# Project page

`index.html` and `static/` are the project page published at https://fairmedfm.github.io/FairMedFM/, together with
the documentation from `docs/` at `/docs/`. The `Pages` workflow builds and deploys both on every push to `main`;
`assemble_site.py` combines them and writes the sitemaps.

Files in `verification/` are published at the site root. Put search engine ownership files there (for Google
Search Console, the "HTML file" method for the URL-prefix property https://fairmedfm.github.io/FairMedFM/). The
32-hex-character `.txt` file is the IndexNow key.

The page design comes from the [Nerfies](https://github.com/nerfies/nerfies.github.io) template (CC BY-SA 4.0).
