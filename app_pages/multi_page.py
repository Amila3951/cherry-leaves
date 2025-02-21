import streamlit as st

class MultiPage:
    """Framework for combining multiple streamlit pages."""

    def __init__(self, app_name) -> None:
        """Initializes multi-page app."""
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🖥️")

    def add_page(self, title, func) -> None:
        """Adds a new page to the app.

        Args:
            title (str): The title of the page.
            func (function): The function that contains the page content.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """Runs the multi-page app."""
        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages,
                                    format_func=lambda page: page['title'])
        page['function']()