"""Legacy Streamlit entry for the formal project frontend.

The production display layer now lives in `services.timeline_api` and serves
`frontend/static` through FastAPI.
"""

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional at import time.
    st = None


def build_app_description() -> str:
    """Return the current frontend status description."""
    return (
        "NewsLine web UI is implemented as a FastAPI-served static frontend. "
        "Run: uvicorn services.timeline_api:app --reload"
    )


def main() -> None:
    """Render the placeholder frontend."""
    description = build_app_description()
    if st is None:
        print(description)
        return

    st.set_page_config(page_title="NewsLine", layout="wide")
    st.title("NewsLine")
    st.caption("正式展示层已迁移到 FastAPI + 静态前端。")
    st.info(description)


if __name__ == "__main__":
    main()
