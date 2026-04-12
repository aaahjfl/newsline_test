"""Streamlit entry placeholder for the formal project frontend."""

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional at import time.
    st = None


def build_app_description() -> str:
    """Return the current frontend status description."""
    return (
        "Formal frontend skeleton is ready. "
        "Connect data pipeline, event discovery, and timeline reasoning modules next."
    )


def main() -> None:
    """Render the placeholder frontend."""
    description = build_app_description()
    if st is None:
        print(description)
        return

    st.set_page_config(page_title="NewsLine", layout="wide")
    st.title("NewsLine")
    st.caption("项目当前处于架构重构与基础骨架搭建阶段。")
    st.info(description)


if __name__ == "__main__":
    main()
