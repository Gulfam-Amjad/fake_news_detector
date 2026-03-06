"""Root Streamlit entrypoint.

This forwards to app/app.py so both commands below run the same app
that loads models/best_model.pkl:

- streamlit run streamlit_app.py
- streamlit run app/app.py
"""

from app.app import main


if __name__ == "__main__":
    main()
