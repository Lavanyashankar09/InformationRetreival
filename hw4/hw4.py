import logging
import re
import sys
from bs4 import BeautifulSoup
from queue import Queue, PriorityQueue
from urllib import parse, request

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')


def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)

def relevance_function(link):
    link_lower = link.lower()
    keywords = ['cs', 'computer', 'programming', 'python', 'java', 'javascript', 'html', 'css']
    keywords_lower = [keyword.lower() for keyword in keywords]
    # Count the occurrences of keywords in the link
    relevance_score = sum(link_lower.count(keyword) for keyword in keywords_lower)
    #print(relevance_score)
    return relevance_score
    
def parse_links_sorted(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            url1 = parse.urljoin(root, link.get('href'))
            yield ((relevance_function(url1), url1), text)

def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))

def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''
    links = get_links(url)
    filtered = []
    for link in links:
        if parse.urlparse(link[0]).hostname != parse.urlparse(url).hostname:
            filtered.append(link)
    return filtered

def crawl(root, wanted_content=[], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement
    queue = Queue()
    queue.put(root)
    visited = []
    extracted = []
    while not queue.empty():
        url = queue.get()
        try:
            req = request.urlopen(url)
            if wanted_content and not any([i.lower() in req.headers['Content-Type'].lower() for i in wanted_content]):
                continue
            html = req.read().decode('utf-8')
            visited.append(url)
            visitlog.debug(url)
            for ex in extract_information(url, html):
                extracted.append(ex)
                extractlog.debug(ex)
            for link, title in parse_links(url, html):
                parsed_link = parse.urlparse(link)
                if link in visited or parsed_link == parse.urlparse(root) or (within_domain and parsed_link.hostname != parse.urlparse(url).hostname):
                    continue
                queue.put(link)
        except Exception as e:
            print(e, url)
    return visited, extracted

def priority_crawl(root, wanted_content=['text/html'], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement
    queue = PriorityQueue()
    queue.put((relevance_function(root), root))
    visited = set()
    extracted = []
    while not queue.empty():
        if len(visited) > 5:
            break
        rank, url = queue.get()
        try:
            req = request.urlopen(url)
            if wanted_content and not any([i.lower() in req.headers['Content-Type'].lower() for i in wanted_content]):
                continue
            html = req.read().decode('utf-8')
            visited.add(url)
            visitlog.debug(url)
            for ex in extract_information(url, html):
                extracted.append(ex)
                extractlog.debug(ex)
            for (rank, link), title in parse_links_sorted(url, html):
                parsed_link = parse.urlparse(link)
                if link in visited or parsed_link == parse.urlparse(root) or (within_domain and parsed_link.hostname != parse.urlparse(url).hostname):
                    continue
                queue.put((rank, link))
        except Exception as e:
            print(e, url)
    return visited, extracted

def extract_information(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''

    # TODO: implement
    results = []
    for match in re.findall(r'\d\d\d-\d\d\d-\d\d\d\d', str(html)):
        results.append((address, 'PHONE', match))
    for match in re.findall('([a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,3})', str(html)): 
        results.append((address, 'EMAIL', match))
    for match in re.findall('[a-zA-Z]+ ?[a-zA-z]+?, [a-zA-Z.]+ [0-9]{5}', str(html)):
        results.append((address, 'ADDRESS', match))
    return results

def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)

def main():
    site = sys.argv[1]
    links = get_links(site)
    writelines('links.txt', links)
    nonlocal_links = get_nonlocal_links(site)
    writelines('nonlocal.txt', nonlocal_links)
    visited, extracted = priority_crawl(site)
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)

if __name__ == '__main__':
    main()