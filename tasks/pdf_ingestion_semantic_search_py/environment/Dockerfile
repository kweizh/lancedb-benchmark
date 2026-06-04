FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        lancedb==0.25.3 \
        pyarrow==17.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        openai==1.54.5 \
        tiktoken \
        pypdf \
        reportlab && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && chmod 0755 /home/user/myproject

# Bake the corpus into the image at build time using reportlab.
# Three PDFs (alpha.pdf, bravo.pdf, charlie.pdf), each with 5 pages of
# distinctive, semantically separable content. The verifier queries
# anchor phrases on specific pages:
#   - bravo  page 3: "the secret password is RAVENCLAW123"
#   - alpha  page 2: cobalt blue butterflies migrating across the Sahara
#   - charlie page 4: procedure to calibrate a quantum gyroscope using lithium niobate
RUN mkdir -p /app/corpus && python3 - <<'PYEOF'
import os
import textwrap

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

OUT_DIR = "/app/corpus"
os.makedirs(OUT_DIR, exist_ok=True)

DOCS = {
    "alpha": [
        # page 1
        (
            "Welcome to the Alpha Manuscript. This is a comprehensive guide to nature, "
            "wildlife, and the ecosystems of the world. The Alpha Manuscript was first "
            "assembled by a group of field biologists in the late 1990s. In the chapters "
            "that follow we explore forest canopies, alpine tundra, deep ocean trenches, "
            "and the wide grasslands of central Asia. We give particular attention to "
            "species whose habits are poorly understood. Throughout, we emphasize the "
            "interdependence of organisms within an ecosystem and the consequences of "
            "habitat fragmentation."
        ),
        # page 2 -- anchor for the butterfly query
        (
            "Chapter Two: The Great Migration. Every spring, vast clouds of cobalt blue "
            "butterflies travel from the foothills of the Atlas Mountains across the "
            "Sahara desert toward the lush Sahel. This astonishing migration of "
            "butterflies across the Sahara has fascinated entomologists for decades. "
            "Their cobalt blue wings reflect ultraviolet light and help regulate body "
            "temperature during the long flight. Researchers have tracked individuals "
            "covering more than two thousand kilometres in a single season. The migration "
            "is timed with the seasonal blooming of acacia trees on the southern edge of "
            "the desert."
        ),
        # page 3
        (
            "Chapter Three: Raptors of the Steppe. The kestrel, the saker falcon, and the "
            "golden eagle are among the apex avian predators of the central Asian steppe. "
            "Each occupies a slightly different niche, ranging from low-altitude rodent "
            "hunting to high-altitude scavenging. The saker falcon in particular has been "
            "the subject of intensive conservation programmes after population declines."
        ),
        # page 4
        (
            "Chapter Four: Marine Life. Coral reefs in the western Pacific harbor "
            "thousands of fish species, each adapted to a specific micro-environment. "
            "We describe parrotfish grazing, cleaner shrimp mutualisms, and the rare "
            "behaviour of the mimic octopus. Reef bleaching events of the past two "
            "decades have reshaped community composition in measurable ways."
        ),
        # page 5
        (
            "Chapter Five: Conclusion. Conservation of biodiversity remains the central "
            "challenge of our century. We close the Alpha Manuscript with a call for "
            "continued field research, citizen science participation, and stronger "
            "protected-area policy at both national and international levels."
        ),
    ],
    "bravo": [
        # page 1
        (
            "Welcome to the Bravo Operations Handbook. This document covers procedures "
            "for maintaining secure access controls inside the company. The handbook is "
            "intended for members of the operations and infrastructure teams. It describes "
            "the workflows used to provision new accounts, rotate credentials, and respond "
            "to access-control incidents."
        ),
        # page 2
        (
            "Chapter Two: Encryption. We use AES-256 encryption with rotating keys for "
            "all data at rest, and TLS 1.3 for data in transit. Key rotation cadence is "
            "every ninety days for production keys and every thirty days for staging keys. "
            "Hardware security modules manage the master keys."
        ),
        # page 3 -- anchor for the password query
        (
            "Chapter Three: Credentials. For the upcoming integration test window, the "
            "secret password is RAVENCLAW123. Please use this password only during the "
            "scheduled test window and do not share it with anyone outside the operations "
            "team. This is the only place in the handbook where the password is recorded. "
            "After the test window closes, the password will be rotated and this section "
            "will be redacted in the next revision."
        ),
        # page 4
        (
            "Chapter Four: Password Policy. Going forward, employee passwords should be "
            "at least sixteen characters long, include mixed case, digits, and symbols, "
            "and must not repeat any password used in the previous five rotations. We "
            "recommend a password manager and we strictly forbid sharing personal "
            "passwords over chat or email."
        ),
        # page 5
        (
            "Chapter Five: Audit Trails. All credential usage is logged centrally and "
            "audited on a weekly basis by the security team. Anomalies are escalated "
            "to the on-call engineer for immediate review and potential account "
            "suspension."
        ),
    ],
    "charlie": [
        # page 1
        (
            "Welcome to the Charlie Engineering Compendium. This volume contains "
            "procedures relating to precision instrumentation and quantum sensors used "
            "throughout the lab. The compendium is updated each year as new measurement "
            "techniques mature. It is intended for graduate students and staff scientists."
        ),
        # page 2
        (
            "Chapter Two: Optics. The refractive index of fused silica varies with "
            "wavelength and temperature. We provide tables of values at the most common "
            "operating wavelengths and a brief discussion of stress-induced birefringence "
            "in optical mounts."
        ),
        # page 3
        (
            "Chapter Three: Classical Mechanics. The pendulum's period depends on the "
            "square root of length divided by the gravitational acceleration. We review "
            "small-angle approximations, parametric resonance, and the practical limits "
            "of mechanical clock accuracy."
        ),
        # page 4 -- anchor for the gyroscope query
        (
            "Chapter Four: Quantum Gyroscope Calibration. The procedure to calibrate a "
            "quantum gyroscope using a lithium niobate substrate begins with thermal "
            "stabilization of the resonator at twenty two degrees Celsius. Apply a "
            "1064 nm laser through the input waveguide, adjust the optical path until "
            "the fringe pattern is centered, and then perform a Sagnac interferometry "
            "sweep across the expected rotation range. The lithium niobate substrate "
            "provides the necessary electro-optic coefficient. Repeat the full "
            "calibration at three rotation rates and record the residuals."
        ),
        # page 5
        (
            "Chapter Five: Future Research. Topological insulators may eventually "
            "replace lithium niobate in next-generation sensors, but the fabrication "
            "techniques are still immature. We close with an outlook on cold-atom "
            "interferometry and chip-scale atomic clocks."
        ),
    ],
}

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN_X = 72
TOP_Y = PAGE_HEIGHT - 72
LINE_HEIGHT = 14
WRAP_WIDTH = 78

for doc_id, pages in DOCS.items():
    pdf_path = os.path.join(OUT_DIR, f"{doc_id}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    for page_index, body in enumerate(pages, start=1):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(MARGIN_X, TOP_Y, f"{doc_id.title()} Document - Page {page_index}")
        c.setFont("Helvetica", 11)
        y = TOP_Y - 2 * LINE_HEIGHT
        for paragraph in body.split("\n"):
            for line in textwrap.wrap(paragraph, width=WRAP_WIDTH):
                c.drawString(MARGIN_X, y, line)
                y -= LINE_HEIGHT
                if y < 72:
                    break
            y -= LINE_HEIGHT // 2
        c.showPage()
    c.save()
    size = os.path.getsize(pdf_path)
    print(f"wrote {pdf_path} ({size} bytes)")
PYEOF

# Sanity check: confirm the PDFs were created and can be parsed by pypdf.
RUN python3 - <<'PYEOF'
import os
from pypdf import PdfReader

names = sorted(os.listdir("/app/corpus"))
assert names == ["alpha.pdf", "bravo.pdf", "charlie.pdf"], names
for name in names:
    reader = PdfReader(f"/app/corpus/{name}")
    assert len(reader.pages) == 5, (name, len(reader.pages))
    text = " ".join((p.extract_text() or "") for p in reader.pages)
    assert text.strip(), f"no extractable text in {name}"
print("corpus sanity OK")
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB PDF ingestion + OpenAI semantic search task. Corpus baked into /app/corpus by reportlab at build time."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3
#           openai==1.54.5 tiktoken pypdf reportlab
#   - /app/corpus/{alpha,bravo,charlie}.pdf  (5 pages each, generated by reportlab)
#   - /home/user/myproject                  (empty project dir for the candidate)
