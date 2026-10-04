"""Create explicitly controlled, label-free monitoring demonstration CSVs."""
import argparse
from pathlib import Path
from credit_risk.monitoring.demo import make_monitoring_demo

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",type=Path,required=True)
    parser.add_argument("--reference-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    print(make_monitoring_demo(args.csv,args.reference_dir,args.output_dir))
