import streamlit as st
from groq import Groq
import json
import os
from io import BytesIO
from markdown import markdown
from weasyprint import HTML, CSS
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit application
st.set_page_config(page_title="Groqbook", page_icon="📚", layout="wide")

# Define GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create a Groq client with the API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Define a custom CSS stylesheet for the application
st.markdown("""
<style>
    // Custom CSS styles for the application
</style>
""", unsafe_allow_html=True)

class GenerationStatistics:
    # Define a class to track generation statistics

    def __init__(self, input_time=0, output_time=0, input_tokens=0, output_tokens=0, total_time=0, model_name="llama3-8b-8192"):
        # Initialize statistical variables
        self.input_time = input_time
        self.output_time = output_time
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_time = total_time
        self.model_name = model_name

    # Define methods to calculate input and output speeds
    def get_input_speed(self):
        return self.input_tokens / self.input_time if self.input_time != 0 else 0

    def get_output_speed(self):
        return self.output_tokens / self.output_time if self.output_time != 0 else 0

    # Define a method to add generation statistics
    def add(self, other):
        if not isinstance(other, GenerationStatistics):
            raise TypeError("Can only add GenerationStatistics objects")

        self.input_time += other.input_time
        self.output_time += other.output_time
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_time += other.total_time

    # Define a method to generate a string representation of the generation statistics
    def __str__(self):
        return (f"## Generation Statistics\n"
                f"- **Model**: {self.model_name}\n"
                f"- **Total Time**: {self.total_time:.2f}s\n"
                f"- **Output Speed**: {self.get_output_speed():.2f} tokens/s\n"
                f"- **Total Tokens**: {self.input_tokens + self.output_tokens}\n")

class Book:
    # Define a class to represent a book

    def __init__(self, structure):
        # Initialize the book with a structure
        self.structure = structure
        self.contents = {title: "" for title in self.flatten_structure(structure)}
        self.placeholders = {title: st.empty() for title in self.flatten_structure(structure)}

    # Define a method to flatten the book structure
    def flatten_structure(self, structure):
        sections = []
        for title, content in structure.items():
            sections.append(title)
            if isinstance(content, dict):
                sections.extend(self.flatten_structure(content))
        return sections

    # Define a method to update the book content
    def update_content(self, title, new_content):
        self.contents[title] += new_content
        self.display_content(title)

    # Define a method to display the book content
    def display_content(self, title):
        if self.contents[title].strip():
            self.placeholders[title].markdown(
                f"""
                <div class='book-section'>
                    <h3>{title}</h3>
                    <div>{self.contents[title]}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

    # Define a method to display the book structure
    def display_structure(self, structure=None, level=2):
        if structure is None:
            structure = self.structure

        for title, content in structure.items():
            if self.contents[title].strip():
                st.markdown(f"<h{level} style='color: var(--text-color);'>{title}</h{level}>", unsafe_allow_html=True)
                self.display_content(title)
            if isinstance(content, dict):
                self.display_structure(content, level + 1)

    # Define a method to generate a markdown content for the book
    def get_markdown_content(self, structure=None, level=1):
        if structure is None:
            structure = self.structure

        markdown_content = ""
        for title, content in structure.items():
            if self.contents[title].strip():
                markdown_content += f"{'#' * level} {title}\n{self.contents[title]}\n\n"
            if isinstance(content, dict):
                markdown_content += self.get_markdown_content(content, level + 1)
        return markdown_content

def create_markdown_file(content: str) -> BytesIO:
    # Create a BytesIO buffer for the markdown content
    markdown_file = BytesIO()
    markdown_file.write(content.encode('utf-8'))
    markdown_file.seek(0)
    return markdown_file

def create_pdf_file(content: str) -> BytesIO:
    # Create a BytesIO buffer for the PDF content
    html_content = markdown(content, extensions=['extra', 'codehilite'])
    styled_html = f"""
    <html>
        <head>
            <style>
                // Custom CSS styles for the PDF
            </style>
        </head>
        <body>
            {html_content}
        </body>
    </html>
    """
    pdf_buffer = BytesIO()
    HTML(string=styled_html).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

def generate_book_structure(prompt: str):
    # Generate the book structure using LLaMa3 on Groq
    completion = groq_client.chat.completions.create(
        model="llama-7.0-alpha-25b",
        messages=[
            {"role": "system", "content": "Write in JSON format:\n\n{\"Title of section goes here\":\"Description of section goes here\",\n\"Title of section goes here\":{\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\",\"Title of section goes here\":\"Description of section goes here\"}}"},
            {"role": "user", "content": f"Write a comprehensive structure, omiting introduction and conclusion sections (forward, author's note, summary), for a long (>300 page) book on the following subject:\n\n<subject>{prompt}</subject>"}
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    usage = completion.usage
    statistics = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name="llama3-70b-8192")

    return statistics, completion.choices[0].message.content

def generate_section(prompt: str):
    # Generate a section of the book using LLaMa3 on Groq
    stream = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an expert writer. Generate a long, comprehensive, structured chapter for the section provided."},
            {"role": "user", "content": f"Generate a long, comprehensive, structured chapter for the following section:\n\n<section_title>{prompt}</section_title>"}
        ],
        temperature=0.3,
        max_tokens=8000,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in stream:
        tokens = chunk.choices[0].delta.content
        if tokens:
            yield tokens
        if x_groq := chunk.x_groq:
            if not x_groq.usage:
                continue
            usage = x_groq.usage
            statistics = GenerationStatistics(input_time=usage.prompt_time, output_time=usage.completion_time, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, total_time=usage.total_time, model_name="llama3-8b-8192")
            yield statistics

def main():
    # Set up Streamlit application
    st.title("📚 Groqbook: Write Full Books using LLaMa3 on Groq")

    with st.sidebar:
        st.header("Generation Statistics")
        stats_placeholder = st.empty()

    col1, col2 = st.columns([3, 1])

    with col1:
        topic_text = st.text_area("What do you want the book to be about?", "", height=100)
        if st.button("Generate Book"):
            if len(topic_text) < 10:
                st.error("Book topic must be at least 10 characters long")
            else:
                generate_book(topic_text, stats_placeholder)

    with col2:
        if 'book' in st.session_state:
            markdown_file = create_markdown_file(st.session_state.book.get_markdown_content())
            st.download_button(
                label='Download as Text',
                data=markdown_file,
                file_name='generated_book.txt',
                mime='text/plain',
                use_container_width=True
            )

            pdf_file = create_pdf_file(st.session_state.book.get_markdown_content())
            st.download_button(
                label='Download as PDF',
                data=pdf_file,
                file_name='generated_book.pdf',
                mime='application/pdf',
                use_container_width=True
            )

    if 'book' in st.session_state:
        st.header("Generated Book Content")
        st.session_state.book.display_structure()

def generate_book(topic_text, stats_placeholder):
    # Generate a book structure and content
    with st.spinner("Generating book structure..."):
        structure_stats, book_structure = generate_book_structure(topic_text)
        stats_placeholder.markdown(str(structure_stats), unsafe_allow_html=True)

    try:
        book_structure_json = json.loads(book_structure)
        book = Book(book_structure_json)
        st.session_state.book = book

        total_stats = GenerationStatistics(model_name="Combined")

        def stream_section_content(sections):
            for title, content in sections.items():
                if isinstance(content, str):
                    with st.spinner(f"Generating content for: {title}"):
                        content_stream = generate_section(f"{title}: {content}")
                        for chunk in content_stream:
                            if isinstance(chunk, GenerationStatistics):
                                total_stats.add(chunk)
                                stats_placeholder.markdown(str(total_stats), unsafe_allow_html=True)
                            elif chunk is not None:
                                st.session_state.book.update_content(title, chunk)
                elif isinstance(content, dict):
                    stream_section_content(content)

        stream_section_content(book_structure_json)
        st.success("Book generation completed!")

    except json.JSONDecodeError:
        st.error("Failed to decode the book structure. Please try again.")

if __name__ == "__main__":
    main()