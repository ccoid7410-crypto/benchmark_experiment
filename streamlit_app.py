from __future__ import annotations

from typing import Dict, List

import streamlit as st


Pattern = List[List[int]]


PATTERNS: Dict[str, Pattern] = {
    "dark": [[0, 0, 0, 0, 0, 0] for _ in range(6)],
    "target": [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    "assist": [
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    "hold": [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ],
    "transfer": [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0],
    ],
    "block": [
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
    ],
    "complete": [[1, 1, 1, 1, 1, 1] for _ in range(6)],
}


PATTERN_TEXT = {
    "dark": "무신호",
    "target": "대상 고정",
    "assist": "공동 개입",
    "hold": "조건 유지",
    "transfer": "다음 작업 진행",
    "block": "위험/금지",
    "complete": "완료 확인",
}


STATIONS = [
    ("intake", "대상 확인", "작업 대상이 무엇인지 먼저 고정한다."),
    ("lift", "공동 운반", "한 에이전트가 단독 처리할 수 없는 상태."),
    ("lock", "안전 잠금", "누군가 조건을 유지해야 다음 작업 가능."),
    ("device", "장치 작동", "Operator가 장치를 켜고 결과물을 만든다."),
    ("delivery", "결과 전달", "결과물을 최종 사용자/구역에 넘긴다."),
]


from maci_components.agent_state import AgentState
from maci_components.mission_state import MissionState


def new_state() -> MissionState:
    state = MissionState()
    state.agents = {
        "A": AgentState("Agent A", "Carrier", "#2f6cbd", "intake"),
        "B": AgentState("Agent B", "Operator", "#b8542c", "device"),
    }
    state.log.append("[000] 환경 생성: 좌표 격자 없이 미션 스테이션과 상호작용 조건만 사용합니다.")
    return state


def get_state() -> MissionState:
    if "mission" not in st.session_state:
        st.session_state.mission = new_state()
    return st.session_state.mission


def log(state: MissionState, text: str) -> None:
    state.log.insert(0, f"[{state.turn:03d}] {text}")
    state.log = state.log[:90]


def station_index(key: str) -> int:
    return [station[0] for station in STATIONS].index(key)


def move(agent: AgentState, station: str, intention: str) -> None:
    agent.station = station
    agent.intention = intention


def set_signal(agent: AgentState, pattern: str, interpretation: str, confidence: int) -> None:
    agent.pattern = pattern
    agent.interpretation = interpretation
    agent.confidence = confidence


def active_flags(state: MissionState) -> List[bool]:
    return [
        state.object_locked,
        state.joint_lift,
        state.safety_locked,
        state.device_done,
        state.delivered,
    ]


def completed(state: MissionState) -> bool:
    return state.delivered


def protocol_score(mode: str) -> int:
    if mode == "6x6 시각 패턴":
        return 92
    if mode == "행동 중심 마커":
        return 68
    return 32


def apply_step(mode: str) -> None:
    state = get_state()
    if completed(state):
        return

    state.turn += 1
    state.mode = mode
    a = state.agents["A"]
    b = state.agents["B"]
    base_conf = protocol_score(mode)

    if mode == "통신 없음" and state.turn % 2 == 1:
        state.idle_turns += 1
        set_signal(a, "dark", "상대 의도 추정 실패", 18)
        set_signal(b, "dark", "다음 조건 확인 불가", 16)
        a.intention = "대상 주변에서 대기"
        b.intention = "장치 앞에서 대기"
        log(state, "통신 없음: 둘 다 다음 조건을 확정하지 못하고 대기합니다.")
        return

    if mode == "행동 중심 마커":
        state.ambiguity += 1
        base_conf -= 18

    if not state.object_locked:
        state.object_locked = True
        state.phase = 1
        move(a, "intake", "작업 대상을 시각적으로 고정")
        move(b, "lift", "상대가 고정한 대상을 보러 이동")
        set_signal(a, "target", "대상 고정으로 해석", base_conf)
        set_signal(b, "target", "같은 작업 대상을 공유", base_conf - 4)
        log(state, "대상 고정 패턴으로 두 에이전트가 같은 물체를 미션 대상으로 맞춥니다.")
        return

    if not state.joint_lift:
        state.joint_lift = True
        state.phase = 2
        move(a, "lift", "단독 처리 불가 상태 표시")
        move(b, "lift", "공동 운반에 합류")
        set_signal(a, "assist", "공동 개입 요청", base_conf)
        set_signal(b, "transfer", "개입 후 다음 단계로 넘김", base_conf - 3)
        log(state, "십자형 패턴이 '도움 필요'가 아니라 실제 공동 개입 조건으로 해석됩니다.")
        return

    if not state.safety_locked:
        state.safety_locked = True
        state.phase = 3
        move(a, "lock", "안전 잠금 조건 유지")
        move(b, "device", "유지 신호를 보고 장치 준비")
        set_signal(a, "hold", "조건 유지 중", base_conf)
        set_signal(b, "transfer", "장치 단계 진입 가능", base_conf - 6)
        log(state, "테두리 패턴이 '잠금 조건 유지'로 해석되어 Operator가 장치 단계로 넘어갑니다.")
        return

    if not state.device_done:
        state.device_done = True
        state.phase = 4
        move(a, "lock", "조건 유지 계속")
        move(b, "device", "장치 작동 완료")
        set_signal(a, "hold", "조건을 계속 유지", base_conf - 5)
        set_signal(b, "complete", "장치 작업 완료", base_conf)
        log(state, "장치 작동이 완료되고 결과물이 생성됩니다.")
        return

    state.delivered = True
    state.phase = 4
    move(a, "delivery", "완료 확인")
    move(b, "delivery", "결과물 전달")
    set_signal(a, "complete", "미션 종료", base_conf)
    set_signal(b, "complete", "미션 종료", base_conf)
    log(state, "두 에이전트가 전체 점등 패턴으로 미션 완료를 확인합니다.")


def inject_perturbation(kind: str) -> None:
    state = get_state()
    state.turn += 1
    a = state.agents["A"]
    b = state.agents["B"]
    if kind == "위험 개입":
        state.repair_count += 1
        set_signal(a, "block", "위험 구역 접근 금지", 82)
        set_signal(b, "block", "작업 중지 후 재계획", 80)
        a.intention = "위험 조건 표시"
        b.intention = "장치 작업 일시 중지"
        log(state, "외부 위험이 발생해 X형 금지 패턴으로 작업을 일시 정지했습니다.")
    else:
        state.ambiguity += 1
        set_signal(a, "target", "대상 후보가 둘로 갈림", 51)
        set_signal(b, "assist", "개입 요청인지 대상 고정인지 재확인 필요", 48)
        log(state, "대상 의미가 흔들려 재확인 상황이 발생했습니다.")


def matrix_markup(pattern: str, color: str) -> str:
    cells = []
    for row in PATTERNS[pattern]:
        for value in row:
            bg = color if value else "#263332"
            glow = f"box-shadow: 0 0 14px {color};" if value else ""
            cells.append(f"<span style='background:{bg};{glow}'></span>")
    return f"""
    <div class="matrix-card">
      <div class="led">{''.join(cells)}</div>
      <div class="caption">{PATTERN_TEXT[pattern]}</div>
    </div>
    """


def environment_svg(state: MissionState) -> str:
    flags = active_flags(state)
    agent_positions = {
        "intake": (105, 160),
        "lift": (290, 160),
        "lock": (475, 160),
        "device": (660, 160),
        "delivery": (845, 160),
    }
    station_nodes = []
    for idx, (key, title, desc) in enumerate(STATIONS):
        x, y = agent_positions[key]
        done = flags[idx]
        active = idx == state.phase and not state.delivered
        fill = "#e7f4ee" if done else "#ffffff"
        stroke = "#1d7c72" if active else "#cfd7cf"
        width = 150
        station_nodes.append(
            f"""
            <rect x="{x - 75}" y="70" width="{width}" height="150" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="{3 if active else 1.5}"/>
            <text x="{x}" y="108" text-anchor="middle" font-weight="700" font-size="17" fill="#1d2525">{title}</text>
            <text x="{x}" y="135" text-anchor="middle" font-size="12" fill="#65706d">{desc}</text>
            <circle cx="{x}" cy="190" r="17" fill="{'#1d7c72' if done else '#edf1eb'}"/>
            <text x="{x}" y="195" text-anchor="middle" font-weight="700" font-size="13" fill="{'#fff' if done else '#65706d'}">{idx + 1}</text>
            """
        )

    connectors = []
    for left, right in zip(STATIONS, STATIONS[1:]):
        x1 = agent_positions[left[0]][0] + 76
        x2 = agent_positions[right[0]][0] - 76
        connectors.append(f'<path d="M{x1} 145 C{x1 + 30} 110 {x2 - 30} 180 {x2} 145" fill="none" stroke="#d9ded6" stroke-width="4"/>')

    agent_nodes = []
    for offset, agent in zip([-18, 18], state.agents.values()):
        x, y = agent_positions[agent.station]
        agent_nodes.append(
            f"""
            <g>
              <circle cx="{x + offset}" cy="250" r="20" fill="{agent.color}" stroke="#fff" stroke-width="4"/>
              <text x="{x + offset}" y="256" text-anchor="middle" font-weight="800" font-size="16" fill="#fff">{agent.name[-1]}</text>
            </g>
            """
        )

    return f"""
    <svg viewBox="0 0 950 300" class="scene-svg" role="img" aria-label="interaction environment">
      <rect x="10" y="20" width="930" height="260" rx="18" fill="#f8faf6" stroke="#d9ded6"/>
      {''.join(connectors)}
      {''.join(station_nodes)}
      {''.join(agent_nodes)}
    </svg>
    """


def css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1380px; padding-top: 1.2rem; }
        .hero {
          background: #fff;
          border: 1px solid #d9ded6;
          border-radius: 8px;
          padding: 18px 20px;
          box-shadow: 0 14px 32px rgba(31,43,38,.08);
        }
        .hero h1 { margin: 0; font-size: 1.8rem; letter-spacing: 0; }
        .hero p { margin: .4rem 0 0; color: #65706d; }
        .scene-svg { width: 100%; height: auto; display: block; }
        .matrix-card {
          border: 1px solid #d9ded6;
          border-radius: 8px;
          padding: 12px;
          background: #fff;
        }
        .led {
          display: grid;
          grid-template-columns: repeat(6, 1fr);
          gap: 6px;
          padding: 8px;
          background: #17201f;
          border-radius: 8px;
          aspect-ratio: 1 / 1;
        }
        .led span { border-radius: 4px; min-height: 18px; }
        .caption { text-align: center; color: #65706d; margin-top: 8px; font-size: .9rem; }
        .agent-card, .metric-card, .reading-card {
          border: 1px solid #d9ded6;
          border-radius: 8px;
          padding: 12px;
          background: #fff;
        }
        .agent-card h4 { margin: 0 0 .35rem; }
        .agent-card p { margin: .15rem 0; color: #65706d; }
        .metric-card small { color: #65706d; }
        .metric-card strong { display: block; margin-top: 4px; font-size: 1.45rem; }
        .reading-card { background: #fbfcf8; min-height: 104px; }
        .stTextArea textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .84rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_agent(agent: AgentState) -> None:
    st.markdown(
        f"""
        <div class="agent-card">
          <h4>{agent.name} · {agent.role}</h4>
          <p><b>현재 위치</b>: {dict((key, title) for key, title, _ in STATIONS)[agent.station]}</p>
          <p><b>의도</b>: {agent.intention}</p>
          <p><b>해석</b>: {agent.interpretation}</p>
          <p><b>신뢰도</b>: {agent.confidence}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="MACI 6x6 Interaction", layout="wide")
    css()

    state = get_state()
    st.markdown(
        """
        <div class="hero">
          <h1>MACI 6x6 Interaction Lab</h1>
          <p>격자/미로 없이, 환경 맥락과 6x6 LED 패턴을 함께 읽어 미션을 수행하는 시뮬레이션</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls, stats = st.columns([1.35, 1])
    with controls:
        mode = st.segmented_control(
            "통신 조건",
            ["6x6 시각 패턴", "행동 중심 마커", "통신 없음"],
            default=state.mode,
        )
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Step", use_container_width=True):
            apply_step(mode)
            st.rerun()
        if c2.button("Run 5", use_container_width=True):
            for _ in range(5):
                apply_step(mode)
                if completed(get_state()):
                    break
            st.rerun()
        if c3.button("위험 개입", use_container_width=True):
            inject_perturbation("위험 개입")
            st.rerun()
        if c4.button("Reset", use_container_width=True):
            st.session_state.mission = new_state()
            st.rerun()

    with stats:
        s1, s2, s3, s4 = st.columns(4)
        metric_values = [
            ("Turn", state.turn),
            ("Ambiguity", state.ambiguity),
            ("Idle", state.idle_turns),
            ("Repair", state.repair_count),
        ]
        for column, (label, value) in zip([s1, s2, s3, s4], metric_values):
            column.markdown(f"<div class='metric-card'><small>{label}</small><strong>{value}</strong></div>", unsafe_allow_html=True)

    st.subheader("Interaction Environment")
    st.markdown(environment_svg(state), unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Mission State")
        flags = active_flags(state)
        for idx, (_, title, desc) in enumerate(STATIONS):
            st.checkbox(f"{title} - {desc}", value=flags[idx], disabled=True, key=f"step_{idx}")

        st.subheader("Event Log")
        st.text_area("event-log", "\n".join(state.log), height=230, label_visibility="collapsed")

    with right:
        st.subheader("6x6 LED Signals")
        a_col, b_col = st.columns(2)
        with a_col:
            st.markdown("**Agent A**")
            st.markdown(matrix_markup(state.agents["A"].pattern, state.agents["A"].color), unsafe_allow_html=True)
        with b_col:
            st.markdown("**Agent B**")
            st.markdown(matrix_markup(state.agents["B"].pattern, state.agents["B"].color), unsafe_allow_html=True)

        st.subheader("Agent Reading")
        a_state, b_state = st.columns(2)
        with a_state:
            render_agent(state.agents["A"])
        with b_state:
            render_agent(state.agents["B"])

        st.subheader("Protocol Reading")
        st.markdown(
            f"""
            <div class="reading-card">
              현재 조건은 <b>{state.mode}</b>입니다. 6x6 모드에서는 문자가 아니라
              모양의 중심성, 테두리, 십자, 방향성, 전체 점등 여부를 환경 단계와 결합해 해석합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
