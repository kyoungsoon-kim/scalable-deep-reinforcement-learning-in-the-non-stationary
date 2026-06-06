# Scalable DRL for the Non-Stationary SCLSP

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Method](https://img.shields.io/badge/Method-Deep%20Controlled%20Learning-success)
![Domain](https://img.shields.io/badge/Domain-Inventory%20Control-9cf)
![Status](https://img.shields.io/badge/Status-Educational%20Scaffold-orange)

> Van Hezewijk, Dellaert, Van Jaarsveld (2025), *Scalable deep reinforcement learning in the
> non-stationary capacitated lot sizing problem*, **IJPE 284:109601** 논문을 재구현합니다.
> 지수적으로 폭발하는 행동공간을 **하위결정 분해(decomposed MDP)** 로 선형화하고,
> **Deep Controlled Learning(DCL)** 으로 정책을 학습해 AMBS 휴리스틱·기존 PPO 구현을 능가합니다.

---

## 🎯 핵심 기여

- **MDP 분해**(§3) — 한 기간의 전체 생산결정을 `(어떤 제품 셋업 → 얼마나 생산)` 하위결정 시퀀스로 쪼개
  행동공간을 **지수 → 선형** 으로 줄입니다. 이 저장소의 중심 구현물입니다.
- **DCL 적용**(§4.1) — 비정상(non-stationary) 다품목 환경에 DCL을 적용하고, 예측정보를 상태에 넣어
  수요가 바뀌어도 **재학습 없이** 정책을 재사용합니다.
- **AMBS 기반 롤아웃 정책**(§4.2, Algorithm 1) — DCL 1세대 롤아웃 정책 겸 벤치마크로 사용합니다.

## 🧩 행동공간 분해 한눈에

```
기간 t 진입
  └─ while 용량 남음:
        1단계 A1 ∈ {p0(중단), p1..pK}   ← 어떤 제품을 셋업/생산할지 (Eq.2)
        2단계 A2 ∈ {q1..q_{C-τ}}         ← 얼마나 생산할지 (Eq.3, 제한 그리드 Eq.6)
  기간 종료 → 수요 관측 → 보유/백오더 비용 부과 (Eq.4)
```

## 📂 파일 구조

```
src/
  demand.py    — 비정상 수요 생성 DGP (van Hezewijk 2023a 근사) + 정상 균등분포 (§5.1)
  env.py       — 분해된 SCLSP MDP 환경 (§3) ★핵심
  rollout.py   — AMBS 기반 롤아웃·벤치마크 정책 (§4.2, Algorithm 1)
  policy.py    — MLP 정책망 + 셋업제품-우선 상태인코딩·행동마스킹 (§4.3, §4.4)
  dcl.py       — Deep Controlled Learning: CRN + Sequential Halving + 지도학습 (§4.1)
  evaluate.py  — 시뮬레이션 + 기간평균비용·Δ_vs_AMBS 지표 (§5.1, Table 3/5)
  utils.py     — config 로딩, 통합 행동공간 (Eq.7)
configs/base.yaml — 모든 하이퍼파라미터 (논문 인용 또는 [UNSPECIFIED] 표기)
notebooks/walkthrough.ipynb — 논문↔코드 연결 + 런타임 점검
PAPER_GUIDE.md     — 논문 섹션별 해설 (educational 모드)
REPRODUCTION_NOTES.md — 미명세·재구성 항목 전부 명시
```

## 🚀 빠른 시작

```bash
# Windows: torch는 py -3.12 에만 설치되어 있습니다
py -3.12 -m pip install -r requirements.txt

# AMBS 벤치마크만 빠르게 평가 (torch 불필요)
py -3.12 src/evaluate.py

# DCL 정책 학습(데모 스케일) 후 AMBS와 비교
py -3.12 src/evaluate.py --train-dcl
```

> 기본값은 **데모 스케일**(노트북 CPU 수분)입니다. 논문 수치(Table 2/6, 클러스터 필요)는
> `configs/base.yaml` 의 `dcl.use_paper_scale: true` 로 전환합니다.

## ⚠️ 재현 한계 (반드시 확인)

이 저장소는 **인용 기반 스캐폴드**이며 비트 단위 재현이 아닙니다. 논문 PDF의 일부 수식(Eq.4)·
Algorithm 1 꼬리·정확한 DGP(van Hezewijk 2023a)는 추출 누락/외부참조라 **재구성·근사**했습니다.
모든 추정은 코드의 `[UNSPECIFIED]`/`[PARTIALLY_SPECIFIED]` 주석과 `REPRODUCTION_NOTES.md` 에
표기했습니다. 정량 재현 전 해당 항목을 검증하십시오.

## 📖 인용

```bibtex
@article{vanhezewijk2025scalable,
  title   = {Scalable deep reinforcement learning in the non-stationary capacitated lot sizing problem},
  author  = {Van Hezewijk, Lotte and Dellaert, Nico and Van Jaarsveld, Willem},
  journal = {International Journal of Production Economics},
  volume  = {284},
  pages   = {109601},
  year    = {2025}
}
```
