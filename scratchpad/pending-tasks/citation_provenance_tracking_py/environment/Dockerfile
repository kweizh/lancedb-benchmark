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
        tiktoken && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject && \
    mkdir -p /app/source_documents && \
    mkdir -p /app/lancedb_data

# Bake the 15 source documents into the image so spans can be resolved offline.
RUN cat >/app/source_documents/photosynthesis.txt <<'EOF'
Photosynthesis is the biological process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. The process takes place primarily in the chloroplasts of plant cells, where the green pigment chlorophyll absorbs photons from sunlight. During the light-dependent reactions, water molecules are split into oxygen, protons, and electrons, releasing molecular oxygen as a byproduct. The electrons are passed along an electron transport chain that pumps protons across the thylakoid membrane, generating a gradient used by ATP synthase to produce ATP. NADPH is also produced, providing reducing power for the next stage. In the Calvin cycle, which occurs in the stroma, ATP and NADPH drive the fixation of carbon dioxide into three-carbon sugars via the enzyme RuBisCO. These sugars are eventually combined into glucose and other carbohydrates, which fuel cellular respiration and growth. The net equation is six carbon dioxide plus six water plus light energy yielding one glucose molecule plus six oxygen molecules.
EOF
RUN cat >/app/source_documents/world_war_one.txt <<'EOF'
World War I, fought between 1914 and 1918, was triggered by a complex web of causes that historians often summarize with the acronym MAIN: Militarism, Alliances, Imperialism, and Nationalism. Decades of arms races, particularly the Anglo-German naval rivalry, created a continent primed for conflict. A tangled system of secret and public alliances bound the great powers into two opposing blocs: the Triple Entente of France, Russia, and Britain, and the Triple Alliance of Germany, Austria-Hungary, and Italy. Imperial competition over colonies in Africa and Asia heightened tensions between European empires. Aggressive nationalism, especially Pan-Slavism in the Balkans and Pan-Germanism in Central Europe, fueled ethnic grievances. The immediate spark was the assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo on June 28, 1914, by Gavrilo Princip, a Bosnian Serb nationalist. Austria-Hungary's ultimatum to Serbia and the subsequent declarations of war activated the alliance system, pulling Germany, Russia, France, and ultimately Britain into a global war that killed more than sixteen million people.
EOF
RUN cat >/app/source_documents/great_wall.txt <<'EOF'
The Great Wall of China is a series of fortifications built across the historical northern borders of China to protect Chinese states and empires from nomadic invaders of the Eurasian Steppe. Construction began as early as the seventh century BCE under various warring states, but the most famous segments were built during the Ming dynasty between 1368 and 1644. The Ming-era wall stretches more than 8,800 kilometers when including all natural barriers and trenches, while archaeological surveys covering all historical periods estimate a total length exceeding twenty-one thousand kilometers. The wall was constructed using locally available materials: rammed earth and wood in the west, and quarried stone and fired bricks closer to the capital. Watchtowers were spaced at intervals to allow rapid signal transmission via smoke and fire. Despite its grandeur, the wall was never an impenetrable barrier; it functioned more as a transportation corridor, customs checkpoint, and early-warning system than as an absolute defensive line.
EOF
RUN cat >/app/source_documents/python_language.txt <<'EOF'
Python is a high-level, interpreted, general-purpose programming language created by Guido van Rossum and first released in 1991. Its design philosophy emphasizes code readability through the use of significant indentation, and its syntax aims to be clear and expressive. Python is dynamically typed and garbage-collected, supporting multiple programming paradigms including structured, object-oriented, and functional programming. The language's standard library is often described as 'batteries included' because it provides modules for a wide variety of tasks ranging from file I/O and networking to data serialization and concurrency. The reference implementation, CPython, is written in C and available on most operating systems. Python is widely used in scientific computing, machine learning, web development, automation, and data analysis. Notable frameworks and libraries include Django and Flask for web applications, NumPy and Pandas for numerical analysis, and PyTorch and TensorFlow for deep learning.
EOF
RUN cat >/app/source_documents/black_holes.txt <<'EOF'
A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape once it crosses the boundary known as the event horizon. Black holes form when massive stars exhaust their nuclear fuel and collapse under their own gravity at the end of their lives. The boundary of a black hole, the event horizon, marks the point of no return; the singularity at the center is a region of theoretically infinite density where the known laws of physics break down. Stellar-mass black holes typically have masses ranging from about three to several tens of solar masses, while supermassive black holes found at the centers of galaxies can range from hundreds of thousands to billions of solar masses. The first direct image of a black hole, taken of the supermassive black hole at the center of the galaxy M87, was released in 2019 by the Event Horizon Telescope collaboration, marking a milestone in observational astronomy.
EOF
RUN cat >/app/source_documents/french_revolution.txt <<'EOF'
The French Revolution, which began in 1789 and ended in the late 1790s, was a period of radical political and societal change in France that profoundly transformed both the country and the broader European political landscape. Triggered by a combination of fiscal crisis, food shortages, and Enlightenment ideas about individual rights and popular sovereignty, the revolution overthrew the absolute monarchy of King Louis XVI and culminated in the rise of Napoleon Bonaparte. Key events included the storming of the Bastille on July 14, 1789, the Declaration of the Rights of Man and of the Citizen, the abolition of feudalism, the execution of the king in 1793, the Reign of Terror under Maximilien Robespierre, and the eventual founding of the First French Republic. The revolution introduced political concepts such as citizenship, equality before the law, and secular government, and it inspired subsequent revolutions across Europe and the Americas.
EOF
RUN cat >/app/source_documents/dna_structure.txt <<'EOF'
DNA, or deoxyribonucleic acid, is the molecule that carries genetic information in all known living organisms and many viruses. Its three-dimensional structure was first proposed in 1953 by James Watson and Francis Crick, building on the X-ray diffraction data of Rosalind Franklin and Maurice Wilkins. DNA is composed of two long polynucleotide chains that wind around each other to form a right-handed double helix. Each strand is built from four nucleotide bases: adenine, thymine, guanine, and cytosine, connected by a sugar-phosphate backbone. Adenine pairs with thymine through two hydrogen bonds, while guanine pairs with cytosine through three hydrogen bonds, giving the molecule both stability and the capacity for reliable replication. The genetic code embedded in the sequence of bases directs the synthesis of proteins via transcription into messenger RNA and translation by ribosomes.
EOF
RUN cat >/app/source_documents/printing_press.txt <<'EOF'
The printing press, invented by Johannes Gutenberg around 1440 in Mainz, Germany, transformed European civilization by making written information abundant, inexpensive, and widely accessible. Before Gutenberg, books were laboriously copied by hand or printed using block printing, which made them rare and prohibitively expensive. Gutenberg's key innovation was movable metal type combined with a press derived from existing wine and olive presses, along with an oil-based ink suitable for metal type. The Gutenberg Bible, printed around 1455, demonstrated the new technology's capability to produce high-quality books at scale. The press fueled the spread of Renaissance humanism, the Protestant Reformation, the Scientific Revolution, and the Enlightenment by enabling rapid dissemination of texts in vernacular languages. Literacy expanded, scientific communities formed across borders, and printed pamphlets shaped public opinion in ways previously impossible.
EOF
RUN cat >/app/source_documents/coral_reefs.txt <<'EOF'
Coral reefs are diverse underwater ecosystems built primarily by colonies of tiny animals called coral polyps that secrete calcium carbonate skeletons. Although coral reefs cover less than one percent of the ocean floor, they host approximately a quarter of all marine species, supporting an estimated four thousand species of fish along with mollusks, crustaceans, sea turtles, and many other organisms. Most reef-building corals live in a mutualistic symbiosis with photosynthetic algae called zooxanthellae, which provide the coral with nutrients in exchange for shelter and access to sunlight. Reefs grow only in shallow, warm, clear, and nutrient-poor tropical seas. They provide enormous ecological and economic value through fisheries, coastal protection from storms, and tourism, but they are under severe threat from climate change, ocean acidification, pollution, overfishing, and destructive fishing practices. Mass coral bleaching events occur when prolonged heat causes corals to expel their zooxanthellae, leaving the coral skeletons white and vulnerable to starvation.
EOF
RUN cat >/app/source_documents/moon_landing.txt <<'EOF'
The Apollo 11 mission, launched by NASA on July 16, 1969, was the first crewed mission to land humans on the Moon. The mission was crewed by commander Neil Armstrong, command module pilot Michael Collins, and lunar module pilot Buzz Aldrin. After a three-day journey, the lunar module Eagle separated from the command module Columbia and descended to the lunar surface on July 20, 1969. Armstrong became the first human to step onto the Moon, famously declaring, 'That's one small step for a man, one giant leap for mankind.' Aldrin joined him on the surface, and together the astronauts spent about two and a quarter hours outside the lander collecting samples, taking photographs, and deploying scientific instruments. The mission fulfilled the goal set by President John F. Kennedy in 1961 of landing a man on the Moon and returning him safely to Earth before the end of the decade. It also marked a defining moment of the Cold War space race between the United States and the Soviet Union.
EOF
RUN cat >/app/source_documents/silk_road.txt <<'EOF'
The Silk Road was a network of ancient trade routes that connected the East and West for centuries, stretching from China across Central Asia to the Mediterranean. The route flourished from around the second century BCE during the Han dynasty until the mid-fifteenth century, facilitating the exchange of silk, spices, tea, porcelain, glassware, precious metals, and ideas. Goods traveled by camel caravan through vast deserts and high mountain passes, with merchants relaying cargo from one oasis city to the next rather than making the entire journey themselves. Beyond commerce, the Silk Road served as a conduit for technology, religion, art, and disease. Buddhism spread from India into China along its branches, while paper-making and gunpowder later traveled westward from China. Maritime spice routes through the Indian Ocean increasingly supplanted the overland routes after the rise of European seafaring powers in the late fifteenth century.
EOF
RUN cat >/app/source_documents/electricity_basics.txt <<'EOF'
Electricity is the set of physical phenomena associated with the presence and motion of electric charge. The two main forms studied in introductory physics are static electricity, which involves charges at rest, and current electricity, which involves charges in motion. Electric current is measured in amperes and is defined as the rate of flow of electric charge past a point in a circuit. Voltage, measured in volts, is the electric potential difference that drives current through a conductor, while resistance, measured in ohms, opposes that flow. Ohm's law states that the current through a conductor between two points is directly proportional to the voltage across the two points and inversely proportional to the resistance. Electricity is generated commercially by converting other forms of energy, such as the chemical energy in fossil fuels, the kinetic energy of wind or flowing water, or the nuclear energy of uranium, into electrical energy via turbines and generators.
EOF
RUN cat >/app/source_documents/internet_history.txt <<'EOF'
The Internet began as a research project funded by the United States Department of Defense in the late 1960s, originally called ARPANET. Its first message was sent in October 1969 between computers at UCLA and the Stanford Research Institute. Throughout the 1970s and 1980s, the network grew as universities and research laboratories connected, and the TCP/IP protocol suite developed by Vint Cerf and Bob Kahn became the standard means of moving data between heterogeneous networks. The system was opened to commercial use in the early 1990s, and the launch of the World Wide Web by Tim Berners-Lee at CERN in 1991 transformed the Internet from a research tool into a mass medium. The introduction of graphical web browsers, broadband connectivity, mobile devices, and social media platforms in the following decades fundamentally reshaped global commerce, communication, politics, and culture.
EOF
RUN cat >/app/source_documents/quantum_mechanics.txt <<'EOF'
Quantum mechanics is the branch of physics that describes the behavior of matter and light at atomic and subatomic scales. It emerged in the early twentieth century from the work of Max Planck, Albert Einstein, Niels Bohr, Werner Heisenberg, Erwin Schrödinger, Paul Dirac, and others. Unlike classical physics, quantum mechanics treats particles such as electrons and photons as having both particle-like and wave-like properties, and it predicts only the probabilities of measurement outcomes rather than deterministic trajectories. Central concepts include quantization of energy, wave-particle duality, the uncertainty principle, and quantum entanglement. The Schrödinger equation governs how the wavefunction of a quantum system evolves over time, while measurement is described by operators acting on that wavefunction. Quantum mechanics underpins modern chemistry, solid-state physics, and emerging technologies such as quantum computing, quantum cryptography, and high-precision atomic clocks.
EOF
RUN cat >/app/source_documents/amazon_rainforest.txt <<'EOF'
The Amazon rainforest is the largest tropical rainforest on Earth, covering roughly 5.5 million square kilometers across nine South American countries, with the majority lying within Brazil. It is home to approximately ten percent of all known species on the planet, including jaguars, river dolphins, harpy eagles, sloths, electric eels, and tens of thousands of plant species. The Amazon River and its tributaries discharge about one-fifth of all the freshwater that enters the world's oceans. The forest plays a critical role in regulating regional and global climate by recycling vast amounts of water through evapotranspiration and by sequestering significant quantities of carbon dioxide. Deforestation driven by cattle ranching, soybean cultivation, illegal logging, mining, and infrastructure projects has accelerated since the late twentieth century, leading to substantial losses of biodiversity, displacement of Indigenous peoples, and concerns about the rainforest reaching a tipping point that could convert large areas to savanna.
EOF

# Bake the seed script that will run at container start (OPENAI_API_KEY is only present at runtime).
RUN cat >/app/_seed.py <<'PYEOF'
import json
import os
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa
from openai import OpenAI

SOURCE_DIR = "/app/source_documents"
DB_PATH = "/app/lancedb_data"
CHUNK_SIZE = 280
CHUNK_OVERLAP = 80


def chunk_text(text: str, size: int, overlap: int):
    spans = []
    step = size - overlap
    i = 0
    n = len(text)
    while i < n:
        start = i
        end = min(i + size, n)
        spans.append((start, end))
        if end == n:
            break
        i += step
    return spans


def embed_batch(client: OpenAI, texts):
    # batch up to ~100 inputs at a time
    out = []
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        # transient retry
        last_err = None
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                out.extend([d.embedding for d in resp.data])
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        if last_err is not None:
            raise last_err
    return out


def main():
    run_id = os.environ.get("ZEALT_RUN_ID")
    if not run_id:
        print("ZEALT_RUN_ID is required", file=sys.stderr)
        sys.exit(2)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required at runtime", file=sys.stderr)
        sys.exit(2)

    table_name = f"chunks_{run_id}"
    marker = f"/app/.seeded_{run_id}"
    if os.path.exists(marker):
        print(f"[seed] table {table_name} already seeded; skipping.")
        return

    rows = []
    chunk_id_counter = 0
    for fname in sorted(os.listdir(SOURCE_DIR)):
        if not fname.endswith(".txt"):
            continue
        doc_id = fname[:-4]
        path = os.path.join(SOURCE_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for span_start, span_end in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_text_str = text[span_start:span_end]
            rows.append(
                {
                    "chunk_id": f"c{chunk_id_counter:05d}",
                    "doc_id": doc_id,
                    "span_start": int(span_start),
                    "span_end": int(span_end),
                    "text": chunk_text_str,
                }
            )
            chunk_id_counter += 1

    print(f"[seed] embedding {len(rows)} chunks with text-embedding-3-small ...")
    client = OpenAI()
    vectors = embed_batch(client, [r["text"] for r in rows])
    assert len(vectors) == len(rows), "embedding count mismatch"
    for r, v in zip(rows, vectors):
        r["vector"] = [float(x) for x in v]

    db = lancedb.connect(DB_PATH)
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("span_start", pa.int64()),
            pa.field("span_end", pa.int64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1536)),
        ]
    )
    if table_name in db.table_names():
        db.drop_table(table_name)
    db.create_table(table_name, data=rows, schema=schema)
    with open(marker, "w") as f:
        f.write("seeded\n")
    print(f"[seed] wrote {len(rows)} chunks to LanceDB table {table_name}")

    # avoid lancedb 0.25.3 SIGABRT during interpreter teardown
    os._exit(0)


if __name__ == "__main__":
    main()
PYEOF

# Entrypoint: idempotently seeds the corpus on first start, then keeps the container alive.
RUN cat >/app/entrypoint.sh <<'EOF'
#!/usr/bin/env bash
set -u
python3 /app/_seed.py
trap : TERM INT
sleep infinity & wait
EOF
RUN chmod +x /app/entrypoint.sh

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB citation-provenance RAG task: 15 baked source docs, runtime-seeded chunks_${ZEALT_RUN_ID} table with real OpenAI text-embedding-3-small vectors, solver calls gpt-4o-mini for cited answers."

CMD ["/app/entrypoint.sh"]

# Container contents:
#   - apt: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - pip3: pytest-json-ctrf, lancedb==0.25.3, pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3, openai==1.54.5, tiktoken
#   - /app/source_documents/<doc_id>.txt — 15 pre-baked source documents
#   - /app/_seed.py — runtime seed script that chunks docs, embeds via text-embedding-3-small, writes chunks_${ZEALT_RUN_ID} table
#   - /app/lancedb_data — LanceDB database directory populated at first container start
#   - /app/entrypoint.sh — idempotent seed runner that runs once then keeps the container alive
#   - /home/user/myproject — empty project directory where the candidate writes solution.py
