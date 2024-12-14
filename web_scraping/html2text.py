import requests
from bs4 import BeautifulSoup

def html2text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    
    heading = [h.get_text() for h in soup.find('h1')][0]
    paragraph_texts = [p.get_text() for p in soup.find_all('p')]

    # Save to a text file
    txt_file = f"{heading}.txt"
    with open(txt_file, "w", encoding="utf-8") as file:
        # Write the heading
        file.write(heading + "\n")
        
        # Write each paragraph
        for paragraph in paragraph_texts:
            file.write(paragraph + "\n")

    print(f"Text file saved as {txt_file}")