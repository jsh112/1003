#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
color_web.py (8색 스와치 버전)
- A_main.py에서 import하여 웹 스와치로 색상을 고름
- 선택 즉시 이벤트로 반환 (빈 문자열("")이면 전체 사용)

사용 예)
from color_web import choose_color_via_web
color = choose_color_via_web()  # 기본 8색
# 또는 color = choose_color_via_web(all_colors=["green","red"])  # 원하는 목록만
"""
import threading
import webbrowser
from typing import List, Dict
from flask import Flask, request, render_template_string

# ====== 전역 상태 ======
_app = None
_server_thread = None
_selected = {"color": None}
_event = threading.Event()

# ====== 팔레트 / 라벨 ======
# 내부 key → HEX
_DEF_COLORMAP = {
    "green":  "#43a047",
    "yellow": "#fdd835",
    "pink":   "#ec407a",
    "red":    "#e53935",
    "purple": "#8e24aa",
    "sky":    "#4fc3f7",
    "blue":   "#1e88e5",
    "orange": "#fb8c00",
}
# 표시용 한글 라벨
_LABELS_KO = {
    "green":"초록색","yellow":"노란색","pink":"핑크색","red":"빨간색",
    "purple":"보라색","sky":"하늘색","blue":"파란색","orange":"주황색",
}
# 밝은 배경: 검정 텍스트(라벨)로
_LIGHT_BG = {"yellow","sky","pink"}

# 기본 8색 (요청 순서)
_DEFAULT_COLORS = ["green","yellow","pink","red","purple","sky","blue","orange"]


def _make_app(all_colors: List[str]):
    app = Flask(__name__)

    TEMPLATE = r"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>색상 선택</title>
      <style>
        :root { --gap: 14px; }
        body { font-family: system-ui, Arial, sans-serif; margin: 24px; }
        h2 { margin: 8px 0 18px; }
        form { margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: var(--gap); max-width: 760px; }
        .swatch { position: relative; width: 100%; aspect-ratio: 1/1; border-radius: 16px; border: 2px solid rgba(0,0,0,.15); cursor: pointer; background: var(--bg, #eee); box-shadow: 0 6px 16px rgba(0,0,0,.08); transition: transform .06s ease; }
        .swatch:hover { transform: translateY(-2px); }
        .label { position: absolute; left: 10px; bottom: 8px; font-weight: 700; letter-spacing: .2px; }
        .light .label { color: #111; text-shadow: 0 1px 2px rgba(255,255,255,.4); }
        .dark  .label { color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,.4); }
        .outline { background: repeating-conic-gradient(#fafafa 0% 25%, #f1f1f1 0% 50%) 0/16px 16px; border: 2px dashed #888; }
        .outline .label { color: #111; }
        .footer { margin-top: 18px; color: #777; font-size: 13px; }
        button { border: none; padding: 0; background: transparent; }
      </style>
    </head>
    <body>
      <h2>색상 스와치에서 선택</h2>
      <form method="post">
        <div class="grid">
          <!-- 전체(필터 없음) -->
          <button class="swatch outline light" name="color" value="" title="전체 (필터 없음)" type="submit">
            <span class="label">전체</span>
          </button>

          <!-- 8색 스와치 -->
          {% for key in colors %}
            {% set bg = colormap.get(key, '#ccc') %}
            {% set is_light = 1 if key in light_bg else 0 %}
            <button class="swatch {{ 'light' if is_light else 'dark' }}" name="color" value="{{key}}" style="--bg: {{bg}}" title="{{labels.get(key, key)}}" type="submit">
              <span class="label">{{ labels.get(key, key) }}</span>
            </button>
          {% endfor %}
        </div>
      </form>
      <p class="footer">선택 즉시 창을 닫고 A_main.py가 계속 진행됩니다.</p>
    </body>
    </html>
    """

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            color = (request.form.get("color") or "").strip().lower()
            _selected["color"] = color
            _event.set()
            return "<script>window.close();</script>선택 완료. 창을 닫아도 됩니다."
        return render_template_string(
            TEMPLATE,
            colors=all_colors,
            colormap=_DEF_COLORMAP,
            labels=_LABELS_KO,
            light_bg=_LIGHT_BG,
        )

    return app


def _run_server(app: Flask, host: str, port: int):
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


def choose_color_via_web(all_colors: List[str] | None = None, defaults: Dict | None = None,
                         host: str = "127.0.0.1", port: int = 5055, auto_open: bool = True) -> str:
    """
    브라우저에서 색상을 클릭 선택하면 문자열 반환.
    - 반환: "green"/"red"/... (빈 문자열 "" 이면 전체 사용)
    - all_colors를 생략하면 기본 8색이 노출됨.
    """
    # 사용할 색 목록 확정 (미지정 시 기본 8색)
    if not all_colors:
        use_colors = list(_DEFAULT_COLORS)
    else:
        # 제공된 목록 중, 우리가 가진 팔레트에 있는 것만 사용 (순서 유지)
        use_colors = [c for c in all_colors if c in _DEF_COLORMAP]

    # 서버 시작
    global _app, _server_thread
    _event.clear()
    _selected["color"] = None

    _app = _make_app(use_colors)
    _server_thread = threading.Thread(target=_run_server, args=(_app, host, port), daemon=True)
    _server_thread.start()

    # 브라우저 열기
    if auto_open:
        try:
            webbrowser.open(f"http://{host}:{port}/")
        except Exception:
            pass

    # 사용자 선택 대기 (무한)
    _event.wait()
    return _selected["color"] or ""
