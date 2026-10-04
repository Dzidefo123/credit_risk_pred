"""Exercise the actual verified model through ASGI and verify local tracking exports."""
import argparse
import json
from pathlib import Path
from fastapi.testclient import TestClient
from credit_risk.api.main import create_app
from credit_risk.monitoring.runner import fresh_directory
from credit_risk.tracking.mlflow import export_to_mlflow
from credit_risk.validation.runner import digest,write_json

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/phase11-integration-001"))
    args=parser.parse_args()
    output=fresh_directory(args.output_dir)
    request={"application_id":"illustrative-001","features":{
        "RevolvingUtilizationOfUnsecuredLines":.2,"age":43.,
        "NumberOfTime30_59DaysPastDueNotWorse":0,"DebtRatio":.2,"MonthlyIncome":5000.,
        "NumberOfOpenCreditLinesAndLoans":3,"NumberOfTimes90DaysLate":0,
        "NumberRealEstateLoansOrLines":1,"NumberOfTime60_89DaysPastDueNotWorse":0,
        "NumberOfDependents":1}}
    before={name:digest(Path("artifacts/phase4-origination-001")/name)
        for name in ("experiment.json","test_consumption.json")}
    responses={}
    with TestClient(create_app()) as client:
        for key,method,path,payload in [("health","GET","/health",None),
            ("score","POST","/score",request),("decision","POST","/decision",request)]:
            response=client.get(path) if method=="GET" else client.post(path,json=payload)
            assert response.status_code==200,response.text
            responses[key]=response.json()
        missing=json.loads(json.dumps(request));missing["features"].pop("MonthlyIncome")
        response=client.post("/decision",json=missing)
        assert response.status_code==200 and response.json()["decision"]=="MANUAL_REVIEW"
        responses["missing_income_decision"]=response.json()
        invalid={**request,"label":1}
        assert client.post("/score",json=invalid).status_code==422
    assert responses["score"]["pd"]==responses["decision"]["pd"]
    exports=[export_to_mlflow("artifacts/phase4-origination-001","origination","configs/tracking.yaml","phase4-frozen-training"),
        export_to_mlflow("artifacts/phase5-validation-001","validation","configs/tracking.yaml","phase5-frozen-validation")]
    from mlflow.tracking import MlflowClient
    for record in exports:
        client=MlflowClient(tracking_uri=record["tracking_uri"])
        run=client.get_run(record["run_id"])
        assert run.info.status=="FINISHED"
        assert len(run.data.metrics)==record["metric_count"]
        assert len(run.data.params)==record["parameter_count"]
    assert all(digest(Path("artifacts/phase4-origination-001")/name)==value for name,value in before.items())
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"integration.json",dict(responses=responses,tracking_exports=exports,
        input_provenance="Invented API examples, not historical applicant outcomes",
        original_final_test_scored=False,original_model_and_test_lock_unchanged=True))
    print(json.dumps({"status":"verified","api_model":responses["health"]["model_version"],
        "tracking_run_ids":[r["run_id"] for r in exports],"output_dir":str(output)}))
