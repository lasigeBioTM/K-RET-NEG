from Bio import Entrez


import scispacy
import spacy
#nlp = en_core_sci_sm.load()
# Load the scispacy model

nlp = spacy.load("en_core_sci_scibert")


def split_abstract_into_sentences(abstract):
    # Process the abstract
    doc = nlp(abstract)
    sentences = [str(sent) for sent in doc.sents]
    return sentences


def identify_biomedical_entities(text):
    # Process the text
    doc = nlp(text)
    biomedical_entities = []
    for ent in doc.ents:
        print(ent.text, ent.label_)
        if ent.label_.startswith("ENTITY_LABEL_PREFIX"):
            biomedical_entities.append(ent.text)
    return biomedical_entities


def search_pubmed(query):
    Entrez.email = 'your_email@example.com'  # Set your email address
    handle = Entrez.esearch(db='pubmed', retmax=500, term=query)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']

def fetch_abstracts(paper_ids):
    Entrez.email = 'your_email@example.com'  # Set your email address
    handle = Entrez.efetch(db='pubmed', id=paper_ids, retmode='xml')
    records = Entrez.read(handle)['PubmedArticle']
    handle.close()
    return records

def contains_keywords(abstract):
    keywords = ['inhibitory', 'reduced', 'inhibited']
    for keyword in keywords:
        if keyword in abstract.lower():
            return True
    return False

def download_abstracts(query):
    paper_ids = search_pubmed(query)
    records = fetch_abstracts(paper_ids)
    relevant_abstracts = [record['MedlineCitation']['Article']['Abstract']['AbstractText'][0] for record in records
                          if 'Abstract' in record['MedlineCitation']['Article'] and contains_keywords(record['MedlineCitation']['Article']['Abstract']['AbstractText'][0])]
    return relevant_abstracts

# Example usage
query = 'protein'  # Set your desired search query
abstracts = download_abstracts(query)
for abstract in abstracts:
    abstract=str(abstract)
    sentences = split_abstract_into_sentences(abstract)
    for sentence in sentences:
        if any(x in sentence for x in ['inhibitory', 'reduced', 'inhibited']):
            entities = identify_biomedical_entities(sentence)
            print(sentence)
            for entity in entities:
                print(entity)
