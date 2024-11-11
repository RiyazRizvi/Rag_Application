import streamlit as st
from api import handle_user_input, clear_conversation
from utils import process_documents, init_session_state
import logging

logging.basicConfig(filename='app_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        st.set_page_config(page_title="Chat with Documents and Images", layout="wide")
        init_session_state()

        st.header("Chat with Documents and Images using LLAMA3 🦙")

        # Sidebar
        with st.sidebar:
            st.title("Document Processing")

            # File uploader
            uploaded_files = st.file_uploader(
                "Upload documents or images",
                accept_multiple_files=True,
                type=['pdf', 'csv', 'xlsx', 'xls', 'pptx', 'docx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
            )

            # Process button
            if st.button("Process Documents"):
                if uploaded_files:
                    with st.spinner("Processing documents..."):
                        if process_documents(uploaded_files):
                            st.session_state.docs_processed = True
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Error processing documents. Check logs for details.")
                else:
                    st.warning("Please upload files before processing.")

            # Separate buttons for clearing conversation and everything
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Clear Conversation"):
                    clear_conversation()

            with col2:
                if st.button("Clear Everything"):
                    st.session_state.memory.clear()
                    st.session_state.conversation = []
                    st.session_state.docs_processed = False
                    st.success("Everything has been cleared!")

            # Status indicator
            st.write("Status:")
            st.write(f"Documents Processed: {'✅' if st.session_state.docs_processed else '❌'}")

        # Chat interface
        user_question = st.chat_input("Ask a question about your documents")
        if user_question:
            handle_user_input(user_question)

    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        st.error("An error occurred in the application. Please check the logs.")

if __name__ == "__main__":
    main()