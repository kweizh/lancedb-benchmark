FROM ubuntu:24.04

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep && \
    pip3 install --break-system-packages \
        pytest-json-ctrf \
        "streamlit==1.40.2" \
        "lancedb==0.25.3" \
        "pyarrow==17.0.0" \
        "numpy==1.26.4" \
        "pandas==2.2.3" \
        "openai==1.54.5" \
        "httpx==0.27.2" \
        "tiktoken" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /app /app/db && chmod -R 0755 /app

# Inline build-time seed script: embeds 50 articles + 5 recipes via the real
# OpenAI text-embedding-3-small API and writes them to two pre-populated
# LanceDB tables under /app/db. The candidate's Streamlit app only needs to
# READ from these tables; it must not re-seed them.
RUN cat > /tmp/seed.py <<'PY'
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

DB_DIR = "/app/db"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# 50 articles. id=7 is the anchor for the quantum-entanglement verification
# query; the surrounding ids are deliberately diverse so the semantic search
# top-1 for "How does quantum entanglement violate Bell's inequality?" is
# unambiguous.
ARTICLES = [
    (1, "C4 Photosynthesis in Tropical Grasses",
     "C4 plants such as maize and sugarcane fix carbon dioxide first into a four-carbon molecule, "
     "oxaloacetate, inside mesophyll cells, then shuttle it to bundle-sheath cells where Rubisco "
     "operates in a high-CO2 environment. This spatial separation of carbon fixation suppresses "
     "photorespiration and gives C4 grasses a strong productivity advantage in hot, sunny climates."),
    (2, "Plate Tectonics and Subduction Zones",
     "Subduction zones are convergent plate boundaries where one lithospheric plate descends beneath "
     "another into the mantle. The downgoing slab releases water that lowers the melting point of the "
     "overlying mantle wedge, producing volcanic arcs such as the Andes and the Japanese islands and "
     "driving most of the planet's largest earthquakes."),
    (3, "The Boltzmann Transport Equation",
     "The Boltzmann transport equation describes the statistical behavior of a dilute gas out of "
     "thermal equilibrium by tracking the evolution of the one-particle distribution function in "
     "position and momentum space. Its collision integral encodes molecular scattering and underlies "
     "modern derivations of fluid dynamics, transport coefficients, and electron transport in solids."),
    (4, "Fermat's Little Theorem in Modular Arithmetic",
     "Fermat's little theorem states that for any prime p and integer a not divisible by p, "
     "a raised to the power (p-1) is congruent to 1 modulo p. The result is a cornerstone of "
     "elementary number theory and provides the basis for primality tests, RSA correctness proofs, "
     "and modular exponentiation algorithms."),
    (5, "Glial Cells and Brain Function",
     "Astrocytes, oligodendrocytes, and microglia, collectively called glia, are non-neuronal cells "
     "in the central nervous system that regulate synaptic homeostasis, myelinate axons, and patrol "
     "the brain for injury and pathogens. Once dismissed as mere support cells, glia are now known "
     "to shape neural circuit development and adult plasticity."),
    (6, "The Hubble Tension in Modern Cosmology",
     "The Hubble tension is the persistent discrepancy between values of the Hubble constant inferred "
     "from the cosmic microwave background and from local distance ladder measurements using Cepheid "
     "variables and Type Ia supernovae. Possible resolutions range from systematic effects to new "
     "physics such as early dark energy or modifications to the standard cosmological model."),
    (7, "Quantum Entanglement and Bell's Inequality",
     "Quantum entanglement describes correlations between particles that cannot be reproduced by any "
     "local hidden-variable theory. John Bell derived an inequality that any local-realist theory must "
     "obey, but experiments on entangled photon pairs prepared in EPR states routinely violate the "
     "Bell inequality. These Bell tests, refined to close the locality and detection loopholes, "
     "confirm that nature is genuinely nonlocal and rule out broad classes of local hidden-variable "
     "explanations of quantum correlations."),
    (8, "CRISPR-Cas9 Genome Editing",
     "CRISPR-Cas9 is an adaptive immune system from bacteria repurposed as a programmable genome "
     "editor. A short guide RNA directs the Cas9 nuclease to a complementary DNA sequence, where it "
     "introduces a double-strand break that the cell then repairs by non-homologous end joining or "
     "homology-directed repair, enabling targeted gene knockout or precise sequence replacement."),
    (9, "Mitochondrial DNA Inheritance",
     "Mitochondrial DNA is a small circular genome inherited almost exclusively from the maternal "
     "lineage in most animals, because sperm mitochondria are tagged for destruction after "
     "fertilization. Mutations in mitochondrial DNA accumulate with age and underlie a distinctive "
     "family of inherited metabolic disorders."),
    (10, "Reinforcement Learning with Policy Gradients",
     "Policy gradient methods train a parameterized policy in reinforcement learning by estimating "
     "the gradient of expected return with respect to the policy's parameters. Algorithms such as "
     "REINFORCE, PPO, and actor-critic variants combine sampled trajectories with a value-function "
     "baseline to reduce variance and stabilize updates."),
    (11, "Public-Key Cryptography and RSA",
     "RSA is a public-key cryptosystem whose security rests on the difficulty of factoring the "
     "product of two large primes. Encryption raises the plaintext to the public exponent modulo "
     "the modulus, and decryption uses a private exponent computed from Euler's totient of the "
     "modulus."),
    (12, "Convolutional Neural Networks for Image Recognition",
     "Convolutional neural networks apply learned filter banks across spatial dimensions of an image, "
     "stacking convolution, nonlinearity, and pooling layers to build translation-equivariant "
     "feature hierarchies. They became the dominant architecture for image classification after "
     "AlexNet's 2012 ImageNet victory."),
    (13, "Climate Feedback Loops",
     "Earth's climate response to forcing is shaped by feedback loops. The ice-albedo feedback "
     "amplifies warming as melting ice exposes darker land and ocean, while the water vapor feedback "
     "increases atmospheric humidity and the greenhouse effect. Cloud feedbacks remain the largest "
     "source of uncertainty in climate sensitivity estimates."),
    (14, "The Standard Model of Particle Physics",
     "The Standard Model describes the known elementary particles, quarks and leptons, and the gauge "
     "bosons that mediate the strong, weak, and electromagnetic interactions, together with the "
     "Higgs boson that gives mass to the fermions and weak bosons through spontaneous electroweak "
     "symmetry breaking. Gravity is not part of the Standard Model."),
    (15, "Black Hole Thermodynamics and Hawking Radiation",
     "Stephen Hawking showed that black holes radiate a nearly thermal spectrum of particles whose "
     "temperature is inversely proportional to the black hole's mass. The result tied black hole "
     "horizon area to entropy and triggered decades of work on the information paradox at the "
     "intersection of general relativity and quantum field theory."),
    (16, "Protein Folding and AlphaFold",
     "The protein folding problem asks how a one-dimensional amino acid sequence determines the "
     "three-dimensional native structure of a protein. AlphaFold, a deep learning system from "
     "DeepMind, achieved near-experimental accuracy on the CASP14 benchmark by combining multiple "
     "sequence alignments with an attention-based geometric model."),
    (17, "Graph Neural Networks for Molecular Property Prediction",
     "Graph neural networks operate on molecular graphs where atoms are nodes and bonds are edges. "
     "Message-passing layers aggregate neighbor information at each atom to predict properties such "
     "as solubility, toxicity, or quantum chemical energies, often outperforming hand-crafted "
     "molecular fingerprints."),
    (18, "Optical Coherence Tomography in Ophthalmology",
     "Optical coherence tomography uses low-coherence interferometry to capture cross-sectional "
     "images of the retina with micrometer resolution. It is the workhorse imaging modality for "
     "diagnosing macular degeneration, diabetic retinopathy, and glaucoma."),
    (19, "Bayesian Inference and Hierarchical Models",
     "Bayesian inference treats unknown parameters as random variables whose posterior distribution "
     "is updated from data using Bayes' rule. Hierarchical models share statistical strength across "
     "groups by linking group-level parameters through a common hyperprior."),
    (20, "Renewable Wind Turbine Aerodynamics",
     "Modern wind turbines extract kinetic energy from the atmosphere using long, slender airfoil "
     "blades whose lift-to-drag ratio is optimized for the local wind regime. Blade pitch and rotor "
     "speed are continuously adjusted to track the optimal tip speed ratio defined by the Betz limit."),
    (21, "Lithium-Ion Battery Chemistry",
     "Lithium-ion batteries store energy by shuttling lithium ions between an intercalation cathode "
     "such as lithium cobalt oxide and a graphite anode. The choice of electrolyte and separator "
     "governs cycle life, fast-charging capability, and thermal runaway behavior."),
    (22, "Topological Insulators and the Quantum Hall Effect",
     "Topological insulators are materials that act as bulk insulators but host conducting surface "
     "states protected by time-reversal symmetry. Their robustness against backscattering is a "
     "direct consequence of nontrivial bulk band topology classified by topological invariants."),
    (23, "Sourdough Bread Fermentation Microbiome",
     "Sourdough starters host a stable consortium of wild yeasts and Lactobacillus bacteria. During "
     "bulk fermentation the lactic and acetic acids produced by the bacteria flavor the dough and "
     "lower its pH while the yeast leaven the loaf with carbon dioxide, giving sourdough its "
     "characteristic tangy crumb and chewy crust."),
    (24, "The Krebs Cycle in Cellular Respiration",
     "The citric acid cycle, also called the Krebs cycle, oxidizes acetyl-CoA into carbon dioxide "
     "while transferring high-energy electrons to NAD+ and FAD. The reduced cofactors then feed the "
     "electron transport chain that drives ATP synthesis through oxidative phosphorylation."),
    (25, "Compiler Optimization Passes",
     "Modern compilers apply a pipeline of optimization passes over an intermediate representation. "
     "Loop-invariant code motion hoists computations out of loops, dead code elimination removes "
     "unreachable instructions, and inlining substitutes a function body at the call site."),
    (26, "The Riemann Hypothesis",
     "The Riemann hypothesis conjectures that every non-trivial zero of the Riemann zeta function "
     "has real part equal to one half. The conjecture controls the distribution of prime numbers and "
     "remains one of the most important open problems in mathematics."),
    (27, "Coral Reef Bleaching",
     "Coral bleaching occurs when stressed coral polyps expel their symbiotic algae, leaving the "
     "white calcium carbonate skeleton exposed. Sustained ocean warming and acidification have "
     "caused mass bleaching events across the Great Barrier Reef and Caribbean reef systems."),
    (28, "Photonic Crystal Fibers",
     "Photonic crystal fibers guide light through a microstructured cladding of air holes arranged "
     "in a periodic lattice. By engineering the lattice, designers can tailor dispersion and "
     "nonlinearity for supercontinuum generation and gas-filled hollow-core transmission."),
    (29, "Antarctic Ice Sheet Dynamics",
     "The Antarctic ice sheet flows from the interior to the coast through fast-moving ice streams "
     "that drain into floating ice shelves. The shelves buttress the upstream ice, and their loss "
     "through ocean-driven melting accelerates grounded ice discharge into the sea."),
    (30, "Functional Programming and Monads",
     "Monads are a structuring abstraction in functional programming that compose computations with "
     "context, such as IO, state, or error handling. They consist of a type constructor and two "
     "operations, return and bind, satisfying the monad laws."),
    (31, "Distributed Consensus and Raft",
     "Raft is a consensus algorithm for replicated state machines that achieves linearizability by "
     "electing a single leader who appends entries to a replicated log. It is designed to be easier "
     "to understand and implement than Paxos while providing equivalent guarantees."),
    (32, "Galactic Rotation Curves and Dark Matter",
     "Spiral galaxies rotate faster at their outskirts than the luminous matter alone would predict, "
     "implying an extended halo of unseen mass. This dark matter explains flat rotation curves, "
     "gravitational lensing, and structure formation in the cosmic web."),
    (33, "Linguistic Phonological Rules",
     "Phonological rules describe how the abstract phonemes of a language are pronounced in context. "
     "Examples include English flapping of /t/ between vowels and Italian raddoppiamento sintattico, "
     "where word-initial consonants gemmate after certain preceding words."),
    (34, "Magnetic Resonance Imaging Physics",
     "Magnetic resonance imaging aligns hydrogen nuclei in a strong static field and perturbs them "
     "with radio-frequency pulses. The relaxation times T1 and T2 of different tissues yield "
     "contrast that distinguishes gray and white matter, cerebrospinal fluid, and pathology."),
    (35, "Volcanic Eruption Style Spectrum",
     "Volcanic eruptions span a continuum from effusive Hawaiian lava flows to explosive Plinian "
     "columns. Magma viscosity, dissolved gas content, and conduit geometry govern whether magma "
     "fragments violently into ash and pumice or flows smoothly to the surface."),
    (36, "Cell-Free Protein Synthesis",
     "Cell-free protein synthesis systems use cellular lysates, ribosomes, and energy regeneration "
     "to translate added DNA or RNA templates outside of living cells. They enable rapid prototyping "
     "of synthetic biology constructs and on-demand production of difficult proteins."),
    (37, "Random Forests and Gradient Boosting",
     "Random forests average the predictions of many decorrelated decision trees grown on bootstrap "
     "samples. Gradient boosting instead fits trees sequentially, each one correcting the residual "
     "errors of the ensemble built so far."),
    (38, "Ocean Thermohaline Circulation",
     "The thermohaline circulation, or global conveyor belt, moves seawater between basins driven by "
     "differences in temperature and salinity. Dense cold salty water sinks at high latitudes in the "
     "North Atlantic and Antarctic, balancing warm surface currents flowing poleward."),
    (39, "Edge Computing and 5G Networks",
     "Edge computing moves data processing close to where data is generated to cut latency and "
     "backhaul costs. Combined with 5G millimeter-wave radios and network slicing, it enables "
     "applications such as augmented reality and autonomous vehicles."),
    (40, "Olfactory Receptor Repertoire",
     "Mammalian olfaction relies on a large family of G-protein coupled olfactory receptors that "
     "recognize volatile odorants. Each sensory neuron expresses a single receptor type, and the "
     "combinatorial activation pattern across many receptors is decoded into perceived smells."),
    (41, "Software Defined Networking",
     "Software-defined networking decouples the network control plane from the data plane and "
     "centralizes routing decisions in a programmable controller. Switches forward packets according "
     "to flow rules pushed down over open protocols such as OpenFlow."),
    (42, "Renewable Solar Photovoltaic Cells",
     "Silicon solar cells convert sunlight to electricity through the photovoltaic effect at a p-n "
     "junction. Anti-reflective coatings, passivated emitters, and bifacial designs have steadily "
     "pushed commercial cell efficiencies past 22 percent."),
    (43, "Information Theory and Channel Capacity",
     "Claude Shannon defined the capacity of a noisy communication channel as the supremum of mutual "
     "information between input and output. The noisy channel coding theorem says reliable "
     "transmission is possible at any rate below capacity through suitable error-correcting codes."),
    (44, "Plant Defense Compounds and Herbivory",
     "Plants synthesize secondary metabolites such as alkaloids, terpenoids, and phenolics that deter "
     "herbivores and pathogens. Some of these compounds are toxic, others are antifeedants, and many "
     "have been repurposed as human medicines, spices, or insecticides."),
    (45, "Anomaly Detection in Time Series",
     "Anomaly detection methods on time series flag observations that deviate from learned normal "
     "behavior. Approaches include statistical models such as ARIMA residuals, isolation forests on "
     "windowed features, and autoencoders that reconstruct the input."),
    (46, "Continental Drift and Wegener's Hypothesis",
     "Alfred Wegener proposed in 1912 that the present continents once formed a single supercontinent "
     "called Pangaea that broke up and drifted apart. The hypothesis was initially rejected because "
     "Wegener lacked a mechanism, but plate tectonics later supplied it."),
    (47, "Reading Frame and the Genetic Code",
     "The genetic code is read in non-overlapping triplets, called codons, that specify amino acids "
     "or stop signals during translation. The choice of reading frame is set by the start codon, and "
     "frame-shift mutations radically change the encoded protein."),
    (48, "Public Health Vaccination Coverage",
     "Herd immunity arises when a sufficiently large fraction of a population is immune to an "
     "infectious disease, breaking sustained transmission. The threshold depends on the basic "
     "reproduction number R0 and the vaccine's efficacy and uptake."),
    (49, "Differential Privacy and Noisy Counts",
     "Differential privacy bounds the influence of any single record on the output of an analysis by "
     "adding calibrated noise, typically Laplace or Gaussian, to query results. It provides a "
     "rigorous, composable privacy guarantee for statistical releases."),
    (50, "Solar Stellar Magnetic Cycles",
     "The Sun's magnetic field reverses polarity roughly every eleven years through a dynamo driven "
     "by differential rotation and convection beneath the photosphere. Sunspot numbers wax and wane "
     "in step with this cycle and are tracked by solar physicists as a proxy for activity."),
]

# 5 recipes. id=102 is the anchor for the vegan-chocolate-cake verification query.
RECIPES = [
    (101, "Classic Spaghetti Carbonara",
     "Spaghetti carbonara is a Roman pasta dish in which al dente spaghetti is tossed with a "
     "sauce of guanciale rendered fat, raw egg yolks, finely grated Pecorino Romano, and plenty of "
     "black pepper. The residual heat of the pasta cooks the eggs into a glossy emulsion without "
     "scrambling them."),
    (102, "Vegan Dark Chocolate Cake",
     "This vegan dark chocolate cake is a rich, flourless plant-based dessert. It contains no eggs "
     "and no dairy, relying on aquafaba and apple sauce for binding and tenderness. The batter is "
     "built from unsweetened cocoa powder, melted dark chocolate, plant milk such as oat or "
     "almond, neutral oil, and sugar. The baked cake is finished with a dark-chocolate ganache made "
     "from coconut cream and bittersweet chocolate, yielding a fudgy, intensely chocolatey crumb."),
    (103, "Thai Green Curry with Tofu",
     "Thai green curry starts with a paste of green chilies, lemongrass, galangal, kaffir lime "
     "leaves, garlic, shallots, and coriander roots pounded into a vibrant emerald base. The paste "
     "is fried in coconut cream to bloom the aromatics, thinned with stock and full-fat coconut "
     "milk, and finished with cubed tofu, bamboo shoots, Thai basil, and a splash of fish sauce or "
     "soy sauce for a vegan version."),
    (104, "Sourdough Bagels with Sesame Crust",
     "These sourdough bagels develop their flavor from a long cold retard of a stiff dough made "
     "with bread flour, active sourdough starter, water, malt syrup, and salt. The shaped bagels "
     "are boiled briefly in a baking-soda bath before being topped with toasted sesame seeds and "
     "baked until deeply browned and chewy."),
    (105, "Beef Bourguignon in Red Wine",
     "Beef bourguignon braises beef chuck in red Burgundy wine with pearl onions, button mushrooms, "
     "lardons, carrots, garlic, and a bouquet garni. The long slow cook in a heavy Dutch oven "
     "renders the beef silky and reduces the braising liquid to a glossy, deeply savory sauce."),
]

def embed_batch(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]

def build_table(db, name, rows_meta, client):
    contents = [r[2] for r in rows_meta]
    # Batch in chunks of 32 to stay well within the API request size.
    vecs = []
    for i in range(0, len(contents), 32):
        chunk = contents[i:i + 32]
        vecs.extend(embed_batch(client, chunk))
    assert all(v.shape == (EMBED_DIM,) for v in vecs), \
        f"Unexpected embedding dim from OpenAI for table {name}"

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
    ])
    data = [
        {"id": int(rows_meta[i][0]),
         "title": rows_meta[i][1],
         "content": rows_meta[i][2],
         "vector": vecs[i].tolist()}
        for i in range(len(rows_meta))
    ]
    if name in db.table_names():
        db.drop_table(name)
    tbl = db.create_table(name, data=data, schema=schema)
    assert tbl.count_rows() == len(rows_meta), \
        f"Seed row count mismatch for {name}: {tbl.count_rows()} vs {len(rows_meta)}"
    return tbl

def main():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set at build time.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    db = lancedb.connect(DB_DIR)

    articles_tbl = build_table(db, "articles", ARTICLES, client)
    cooking_tbl = build_table(db, "cooking", RECIPES, client)

    # Sanity-check the two verifier ground-truth queries:
    # 1) articles -> top-1 must be id=7 (quantum entanglement)
    q1 = "How does quantum entanglement violate Bell's inequality?"
    qv1 = embed_batch(client, [q1])[0].tolist()
    hits1 = articles_tbl.search(qv1).limit(3).to_list()
    top_id1 = int(hits1[0]["id"])
    print(f"[seed] articles top-3 for {q1!r}: {[(int(h['id']), h.get('title')) for h in hits1]}")
    assert top_id1 == 7, (
        f"Build-time sanity check failed: expected articles top-1 id=7 for the "
        f"entanglement query, got id={top_id1}."
    )

    # 2) cooking -> top-1 must be id=102 (vegan dark chocolate cake)
    q2 = "rich plant-based chocolate cake without eggs"
    qv2 = embed_batch(client, [q2])[0].tolist()
    hits2 = cooking_tbl.search(qv2).limit(3).to_list()
    top_id2 = int(hits2[0]["id"])
    print(f"[seed] cooking top-3 for {q2!r}: {[(int(h['id']), h.get('title')) for h in hits2]}")
    assert top_id2 == 102, (
        f"Build-time sanity check failed: expected cooking top-1 id=102 for the "
        f"vegan-chocolate-cake query, got id={top_id2}."
    )

    print("[seed] both ground-truth queries verified at build time.")
    # Work around lancedb 0.25.3 occasional SIGABRT during interpreter shutdown
    # in write-heavy build-time seed scripts.
    os._exit(0)

main()
PY

RUN python3 /tmp/seed.py && rm -f /tmp/seed.py

LABEL Author="generate-harbor-tasks"
LABEL Description="Streamlit + LanceDB explorer environment with 50 articles + 5 recipes pre-embedded via real OpenAI text-embedding-3-small."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, streamlit==1.40.2, lancedb==0.25.3, pyarrow==17.0.0,
#           numpy==1.26.4, pandas==2.2.3, openai==1.54.5, tiktoken
#   - /app: candidate project root (where app.py is to be written)
#   - /app/db: pre-populated LanceDB with two tables:
#       * articles (50 rows, id 1..50; id=7 is the quantum-entanglement anchor)
#       * cooking  (5 rows, id 101..105; id=102 is the vegan dark chocolate cake anchor)
#     vectors are 1536-d float32 from OpenAI text-embedding-3-small.
#   - OPENAI_API_KEY: real key consumed at build (seed) and runtime (candidate + verifier)
