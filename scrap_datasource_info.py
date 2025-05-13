
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time


# Define your questions
QUESTIONS = [
    "Is the data source described on this website a primary source? Indicate 'primary' if the data source does not integrate other databases or sources otherwise available on another website for download. If it is not primary indicate 'secondary'.",
    "If the source is secondary, list the data sources it uses and the corresponding websites where to find the sources, else return 'NaN'.",
    "If this data source has been published, indicate the publication reference preferrentially the corresponding PMID.",
    "Indicate the Biolink categories represented in this datasource using Named Entities from the categories names and definitions in this file: https://github.com/biolink/biolink-model/blob/master/src/biolink_model/schema/biolink_model.yaml.",
    "Indicate the Biolink predicates represented in this datasource using Named Predicates from the categories names and definitions in this file: https://github.com/biolink/biolink-model/blob/master/src/biolink_model/schema/biolink_model.yaml.",
    "Indicate the Biolink qualifiers on predicates represented in this datasource using Named qualifiers from the categories names and definitions in this file: https://github.com/biolink/biolink-model/blob/master/src/biolink_model/schema/biolink_model.yaml.",
    "On which Biolink category is the data source centric? ",
    "What is the datasource liscence? Give the liscence name.",
    "Is the documentation on this data source enough to reuse the information?",
    "Is the datasource entirely downloadable? Indicate 'entirely', 'partially', or 'no'.",
    "Has this source an API service?",
    "Can all data from this source be retrieved using the API service? Indicate 'entirely', 'partially', or 'no'.",
    "Does this source represents (1) results from consumed or curated datapoints (publications or models on multiple datapoints) or (2) single data points (dataset)?  If (1) indicate 'KS'; if (2) indicate 'dataset'.",
    "How often is this data source used in the field? Indicate a normalized number of publications between 0 (no other publication mentioning it) and 100 (all publications requiring any info on the categories or predicates refer to this source)",
    "What do researchers say about how they trust this source for biomedical research on social networks and blogs? Indicate a number between 0 (no trust) and 1 (perfect trust) from the winning point of view or -1 if no information about user trust was available.",
    "What information is unique about this source compared to similar sources? Provide a 1 sentence answer."
]

def scrape_page(url,llm_window):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return text[:llm_window]  # Truncate to fit LLM context window


def generate_prompt(page_text, questions,max_new_tokens,temperature):
    prompt = (
        "You are an expert web analyst. Given the following website content (start with the webpage provided from this website), answer these questions:\n"
        f"{page_text}\n\n"
    )
    for i, q in enumerate(questions, 1):
        prompt += f"Q{i}: {q}\nA{i}:"

    return prompt

def get_internal_links(base_url, html):
    soup = BeautifulSoup(html, 'html.parser')
    base_domain = urlparse(base_url).netloc
    links = set()
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc == base_domain:
            links.add(full_url.split('#')[0])  # Remove fragment
    return links

def crawl_website(start_url, max_pages=100):
    visited = set()
    to_visit = set([start_url])
    all_pages = []

    with tqdm(total=max_pages, desc="Crawling pages") as pbar:
        while to_visit and len(visited) < max_pages:
            url = to_visit.pop()
            if url in visited:
                continue
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    html = response.text
                    all_pages.append(url)
                    new_links = get_internal_links(url, html)
                    to_visit.update(new_links - visited)
                visited.add(url)
                pbar.update(1)
            except Exception as e:
                visited.add(url)
                pbar.update(1)
                continue
    return all_pages

if __name__ == "__main__":
    # Load Me-LLaMA model and tokenizer (replace with the correct model path or Hugging Face repo)
    model_name = "nlpie/Llama2-MedTuned-13b"  # Example; use the actual model you have access to
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    
    # parameters:
    llm_window = 4000
    max_new_tokens=512
    temperature=0.2
    max_pages=100
    
       
    # Read the initial URL from Excel
    df = pd.read_excel(r".\data\data_sources_surveyed.xlsx")
    start_url = df["Data source URL"][0]  # Assuming the first row contains the starting page

    # Crawl the website to get all internal pages (limit to 100 for safety)
    all_urls = crawl_website(start_url, max_pages)

    if len(all_urls)>0:
        results = []
        for url in tqdm(all_urls, desc="Processing pages"):
            page_text = scrape_page(url,llm_window)
            row = {'url': url}
            if not page_text:
                for i, q in enumerate(QUESTIONS):
                    row[f'Answer_{i+1}'] = "Error: Could not retrieve page content."
            else:
                for i, question in enumerate(QUESTIONS):
                    prompt = generate_prompt(page_text, question,max_new_tokens,temperature)
                    # Generate response with Me-LLaMA
                    response = llm_pipeline(prompt, max_new_tokens=max_new_tokens, temperature=temperature)[0]['generated_text']
                    # Extract only the answers (optional: post-process as needed)
                    answer  = response[len(prompt):].strip()
                    row[f'Answer_{i+1}'] = answer
                    time.sleep(1)  
            results.append(row)

        # Prepare column names
        columns = ['url'] + [f'Answer_{i+1}' for i in range(len(QUESTIONS))]
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_excel("output_with_answers.xlsx", index=False)