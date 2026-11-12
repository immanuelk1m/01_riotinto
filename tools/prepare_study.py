"""검증한 연구 명세에서 별도 실행 대기열을 만든다. 기존 conf/base 파일은 보존한다."""
import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.studies import load_study, select_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("conf/studies/p0_p1_v1.yml"))
    parser.add_argument("--phase", help="명세에 정의한 단계. 생략하면 전체")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = load_study(args.study)
    specs = select_experiments(study, args.phase)
    output = args.output or Path("data/07_reporting/studies") / study["study_id"] / "queue.yml"
    if output.resolve() == Path("conf/base/parameters_experiments.yml").resolve():
        raise ValueError("기존 실행 YAML 은 덮어쓰지 않는다. 별도 --output 경로를 사용한다")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({"experiments": specs}, sort_keys=False, allow_unicode=True))
    print(f"{study['study_id']}: {len(specs)} experiments -> {output}")


if __name__ == "__main__":
    main()
