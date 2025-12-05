# Citation Formatting Skill

## Purpose

Provide guidance on formatting citations and references from indexed papers for academic use.

## When to Use

Invoke this skill when:
- Generating citation lists from search results
- Formatting references for academic writing
- Creating bibliographies
- Exporting to citation managers

## Citation Styles

### APA 7th Edition

**Journal Article:**
```
Author, A. A., & Author, B. B. (Year). Title of article. Journal Name, Volume(Issue), pages. https://doi.org/xxxxx
```

**Book:**
```
Author, A. A. (Year). Title of book. Publisher.
```

**Book Chapter:**
```
Author, A. A. (Year). Title of chapter. In E. E. Editor (Ed.), Title of book (pp. xx-xx). Publisher.
```

**Thesis:**
```
Author, A. A. (Year). Title of thesis [Doctoral dissertation, University Name]. Database Name.
```

### MLA 9th Edition

**Journal Article:**
```
Author. "Title of Article." Journal Name, vol. X, no. X, Year, pp. xx-xx.
```

**Book:**
```
Author. Title of Book. Publisher, Year.
```

### Chicago (Author-Date)

**Journal Article:**
```
Author, First. Year. "Title of Article." Journal Name Volume (Issue): pages.
```

**Book:**
```
Author, First. Year. Title of Book. Place: Publisher.
```

### BibTeX

**Journal Article:**
```bibtex
@article{citekey,
  author = {Author, First and Author, Second},
  title = {Title of Article},
  journal = {Journal Name},
  year = {2023},
  volume = {42},
  number = {3},
  pages = {100--120},
  doi = {10.1000/example}
}
```

**Book:**
```bibtex
@book{citekey,
  author = {Author, First},
  title = {Title of Book},
  publisher = {Publisher Name},
  year = {2023},
  address = {City}
}
```

## Field Mapping

### From Zotero Metadata to Citation

| Zotero Field | Citation Element |
|--------------|------------------|
| authors | Author names |
| title | Title |
| publicationTitle | Journal/Book title |
| date / publication_year | Year |
| volume | Volume |
| issue | Issue/Number |
| pages | Pages |
| doi | DOI |
| isbn | ISBN |
| publisher | Publisher |
| url | URL (if no DOI) |

### Author Name Formatting

**APA/Chicago:**
- Last, F. M.
- Multiple: Last, F. M., & Last, F. M.
- 3+ in APA: Last, F. M., Last, F. M., ... & Last, F. M.

**MLA:**
- Last, First Middle.
- Multiple: Last, First, and First Last.

### Handling Missing Fields

| Missing Field | Action |
|---------------|--------|
| DOI | Include URL if available |
| Volume/Issue | Omit, include pages if available |
| Pages | Omit, include article number if applicable |
| Publisher | Mark as [Publisher unknown] |
| Year | Use (n.d.) for no date |

## Citation Generation from Search Results

### Workflow

1. Extract paper_id from search result
2. Retrieve full metadata from papers.json
3. Format according to requested style
4. Include all available fields

### Example Output

**Search result to APA:**

From:
```json
{
  "title": "Network Analysis Methods",
  "authors": [{"first_name": "John", "last_name": "Smith"}],
  "publication_year": 2023,
  "journal": "Scientometrics",
  "volume": "128",
  "issue": "3",
  "pages": "100-120",
  "doi": "10.1000/example"
}
```

To:
```
Smith, J. (2023). Network analysis methods. Scientometrics, 128(3), 100-120. https://doi.org/10.1000/example
```

## In-Text Citations

### APA

- One author: (Smith, 2023)
- Two authors: (Smith & Jones, 2023)
- Three+ authors: (Smith et al., 2023)
- With page: (Smith, 2023, p. 15)
- Multiple works: (Smith, 2023; Jones, 2022)

### MLA

- One author: (Smith 15)
- Two authors: (Smith and Jones 15)
- Three+ authors: (Smith et al. 15)

### Chicago

- Same as APA for author-date style

## Quality Checks

### Citation Completeness

- [ ] All authors included
- [ ] Year present
- [ ] Title accurate
- [ ] Source identified
- [ ] DOI/URL for accessibility

### Common Errors

1. Missing volume/issue for journals
2. Incorrect author order
3. Title capitalization wrong for style
4. DOI format inconsistent
5. Missing required fields for style
