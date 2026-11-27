# Citations and acknowledgments

Scientific work based on GREEN should acknowledge both the software and the
scientific methods on which the calculation relies.

Start from the independent citation authority:

- Policy and generated guidance:
  <https://github.com/Green-Phys/green-citations/blob/main/CITATION.md>
- Bibliographic metadata:
  <https://github.com/Green-Phys/green-citations/blob/main/CITATION.cff>
- Generated BibTeX:
  <https://github.com/Green-Phys/green-citations/blob/main/references.bib>

The website mirrors the generated guidance for human readers at
<https://green-phys.org/docs/publications/> and provides acknowledgment text at
<https://green-phys.org/docs/acknowledgment/>. Those pages do not own the
publication metadata or component-to-publication mappings.

Use only approved mappings in `CITATIONS.yaml`. A mapping marked
`review-required`, or a method absent from the mapping, is not permission to
invent a citation choice. Inspect the relevant theory pages for candidate
original literature behind the methods and algorithms actually used, such as
GF2, GW, self-energy embedding, convergence acceleration, imaginary-time
representations, or analytic continuation, then surface the choice for
researcher review.

Do not infer that one software citation replaces method citations. Conversely,
do not produce an exhaustive bibliography of every transitive dependency.
Identify the calculation's actual methods and present the canonical references
to the researcher for final scholarly judgment.

Release announcements and theory pages may provide context, but do not override
the independent policy. When they differ from `green-citations`, report the
drift instead of silently choosing one.

