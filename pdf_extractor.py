from pyzerox import zerox
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import json
import asyncio
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from anthropic import Anthropic
import csv
import streamlit as st
import pandas as pd
import shutil
import logging

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Changed to gpt-4o-mini

# Initialize Claude client with API key from .env
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

anthropic = Anthropic(api_key=anthropic_api_key)

# Custom system prompt for detecting multiple resumes
SPLIT_PROMPT = """
You are a document analyzer. Your task is to:
1. Determine if this document contains multiple resumes
2. If it does, identify where each resume starts and ends
3. Return the count of resumes found

Look for clear indicators like:
- New contact information sections
- Multiple different names
- Distinct education/experience sections
- Page breaks between resumes

Return your response as a JSON with:
- "multiple_resumes": boolean
- "resume_count": number
"""

# Custom prompt template for resume parsing
RESUME_PROMPT = """
You are a resume parser. Extract the following information from the resume in a structured format:
- Full Name
- Email
- Phone Number
- Education (including institution, degree, graduation year)
- Work Experience (including company names, positions, dates)
- Skills
- Certifications (if any)

Resume content:
{text}

Format the output as a JSON object with these fields. Ensure the output is valid JSON format.
"""

prompt = ChatPromptTemplate.from_template(RESUME_PROMPT)

async def check_multiple_resumes(text: str) -> Dict:
    """Check if the document contains multiple resumes."""
    messages = [{"role": "user", "content": SPLIT_PROMPT + "\n\n" + text}]
    response = llm.invoke(messages)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"multiple_resumes": False, "resume_count": 1}

async def split_resumes(text: str) -> List[str]:
    """Split text into individual resumes using Zerox's capabilities."""
    try:
        # Use Zerox to identify document boundaries
        split_result = await zerox(
            text=text,
            model="gpt-4o-mini",  # Changed to gpt-4o-mini
            custom_system_prompt="Split this document into individual resumes. Return each resume as a separate section.",
            return_segments=True
        )
        return split_result if isinstance(split_result, list) else [text]
    except Exception as e:
        print(f"Error splitting resumes: {str(e)}")
        return [text]

async def process_single_resume(text: str) -> Dict:
    """Process a single resume text and extract information."""
    try:
        messages = prompt.format_messages(text=text)
        response = llm.invoke(messages)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            print("Error parsing JSON response")
            return {}
            
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        return {}

async def process_pdf_file(file_path: str, output_dir: str) -> List[Dict]:
    """Process a PDF file that might contain multiple resumes."""
    try:
        # Extract text from PDF using Zerox
        extracted_text = await zerox(
            file_path=file_path,
            model="gpt-4o-mini",  # Changed to gpt-4o-mini
            max_tokens=4096,
            output_dir=output_dir,
        )
        
        # Check if the document contains multiple resumes
        check_result = await check_multiple_resumes(extracted_text)
        
        if check_result.get("multiple_resumes", False):
            # Split the document into individual resumes
            resume_texts = await split_resumes(extracted_text)
        else:
            resume_texts = [extracted_text]
        
        # Process each resume
        results = []
        for idx, resume_text in enumerate(resume_texts, 1):
            result = await process_single_resume(resume_text)
            result["resume_index"] = idx
            results.append(result)
        
        return results
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [{}]

async def process_multiple_files(resume_directory: str, output_dir: str) -> Dict[str, List[Dict]]:
    """Process multiple PDF files from a directory."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files in the directory
    pdf_files = [str(f) for f in Path(resume_directory).glob("*.pdf")]
    
    results = {}
    # Process files with progress bar
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        file_results = await process_pdf_file(pdf_file, output_dir)
        results[pdf_file] = file_results   
    return results

def json_to_csv(json_content: str, output_dir: str) -> None:
    """Convert JSON string content to CSV file with proper None handling."""
    try:
        # Parse the JSON string to Python object
        parsed_json = json.loads(json_content)
        
        # Create CSV file path
        csv_file = Path(output_dir) / "parsed_resumes.csv"
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(['Name', 'Email', 'Mobile', 'Experience', 'Skills'])
            
            # Write data rows with None handling
            for resume in parsed_json:
                # Handle None values for skills
                skills = resume.get('skills', [])
                if skills is None:
                    skills = []
                skills_str = ';'.join(str(skill) for skill in skills if skill is not None)
                
                # Handle None values for experience
                experience = resume.get('experience', '')
                if experience is not None:
                    experience = str(experience).replace('\n', ' ')
                else:
                    experience = ''
                
                # Handle None values for all fields
                writer.writerow([
                    str(resume.get('name', '') or ''),
                    str(resume.get('email', '') or ''),
                    str(resume.get('mobile', '') or ''),
                    experience,
                    skills_str
                ])
        
        print(f"Successfully converted JSON to CSV at: {csv_file}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {str(e)}")
        logging.error(f"JSON content causing error: {json_content}")
    except Exception as e:
        print(f"Error converting to CSV: {str(e)}")
        logging.error(f"Resume data causing error: {parsed_json if 'parsed_json' in locals() else 'JSON not parsed'}")

async def convert_md_to_json(file_path: str, output_dir: str) -> None:
    """Convert MD file to JSON using Claude API and save as CSV."""
    try:
        # Read the markdown file
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
            
        # Define the extraction prompt
        extraction_prompt = f"""
        You are a resume parser. The markdown content contains multiple candidate resumes. For each candidate, extract:
        - Full Name
        - Email
        - Mobile/Phone Number
        - Experience
        - Skills

        Format the output as a JSON array where each object represents a candidate with these fields:
        - "name": string
        - "email": string
        - "mobile": string
        - "experience": string
        - "skills": array of strings

        Example format:
        [
            {{
                "name": "John Doe",
                "email": "john@email.com",
                "mobile": "+1-555-0123",
                "experience": "5 years",
                "skills": ["Python", "JavaScript", "AWS"]
            }}
        ]
        
        Here's the Markdown content to parse:
        {md_content}

        Return only the JSON array, ensure it's valid JSON format.
        """
        
        # Call Claude API
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": extraction_prompt
            }]
        )
        
        json_content = response.content[0].text.strip()
        print("Extracted JSON content:", json_content)
        
        # Convert JSON to CSV right here
        json_to_csv(json_content, output_dir)
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

async def main():
    st.title("📄 Resume Parser")
    st.write("Upload PDF resumes to extract information into a structured format.")

    # Define directories
    resume_directory = "./resumes"
    output_dir = "./output_results"

    # Create directories if they don't exist
    os.makedirs(resume_directory, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )

    if uploaded_files:
        if st.button("🔍 Process Resumes"):
            try:
                # Clear existing files in directories
                st.info("Clearing existing files...")
                for dir_path in [resume_directory, output_dir]:
                    for file_path in Path(dir_path).glob("*"):
                        if file_path.is_file():
                            file_path.unlink()
                
                # Save uploaded files to resume directory
                with st.spinner("Saving uploaded files..."):
                    for uploaded_file in uploaded_files:
                        file_path = Path(resume_directory) / uploaded_file.name
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())

                # Process the files
                with st.spinner("Processing resumes..."):
                    # Process PDFs to MD files
                    results = await process_multiple_files(resume_directory, output_dir)
                    
                    # Process the generated MD files to JSON
                    md_files = list(Path(output_dir).glob("*.md"))
                    
                    if not md_files:
                        st.error("❌ No markdown files were generated. Processing may have failed.")
                        return
                    
                    for md_file in md_files:
                        await convert_md_to_json(str(md_file), output_dir)
                    
                    # Calculate totals
                    total_resumes = sum(len(file_results) for file_results in results.values())        
                 
                    # Display CSV if available
                    csv_path = Path(output_dir) / "parsed_resumes.csv"
                    if csv_path.exists():
                        st.write("### 📊 Results:")
                        df = pd.read_csv(csv_path)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            csv_content = f.read()
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_content,
                            file_name="parsed_resumes.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                raise e

# Add custom CSS for better styling
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1v0mbdj.etr89bj1 {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())