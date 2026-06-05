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
        cohere && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

# Bake the 90-row multilingual corpus into the image deterministically.
RUN python3 - <<'PY'
import json
import os

concepts = [
    # 0
    {
        "en": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "es": "La Torre Eiffel es una torre de celosía de hierro forjado situada en París, Francia.",
        "fr": "La Tour Eiffel est une tour en treillis de fer forgé située à Paris, en France.",
    },
    # 1
    {
        "en": "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
        "es": "El Monte Everest es la montaña más alta de la Tierra, situada en el Himalaya.",
        "fr": "Le mont Everest est la plus haute montagne de la Terre, située dans l'Himalaya.",
    },
    # 2
    {
        "en": "The Amazon River is the largest river in South America by water volume.",
        "es": "El río Amazonas es el río más caudaloso de América del Sur por su volumen de agua.",
        "fr": "L'Amazone est le plus grand fleuve d'Amérique du Sud par son volume d'eau.",
    },
    # 3
    {
        "en": "The Great Wall of China is an ancient fortification stretching thousands of kilometers.",
        "es": "La Gran Muralla China es una fortificación antigua que se extiende miles de kilómetros.",
        "fr": "La Grande Muraille de Chine est une fortification antique s'étendant sur des milliers de kilomètres.",
    },
    # 4
    {
        "en": "The Pacific Ocean is the largest and deepest ocean on the planet.",
        "es": "El océano Pacífico es el océano más grande y profundo del planeta.",
        "fr": "L'océan Pacifique est l'océan le plus grand et le plus profond de la planète.",
    },
    # 5
    {
        "en": "The Sahara Desert in northern Africa is the largest hot desert in the world.",
        "es": "El desierto del Sahara en el norte de África es el desierto cálido más grande del mundo.",
        "fr": "Le Sahara, en Afrique du Nord, est le plus grand désert chaud du monde.",
    },
    # 6
    {
        "en": "The polar bear is a large white bear native to the Arctic Circle.",
        "es": "El oso polar es un gran oso blanco originario del Círculo Polar Ártico.",
        "fr": "L'ours polaire est un grand ours blanc originaire du cercle arctique.",
    },
    # 7
    {
        "en": "Coffee beans are roasted seeds used to brew the popular drink coffee.",
        "es": "Los granos de café son semillas tostadas que se utilizan para preparar la popular bebida del café.",
        "fr": "Les grains de café sont des graines torréfiées utilisées pour préparer la boisson populaire qu'est le café.",
    },
    # 8
    {
        "en": "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "es": "La fotosíntesis es el proceso mediante el cual las plantas verdes convierten la luz solar en energía química.",
        "fr": "La photosynthèse est le processus par lequel les plantes vertes convertissent la lumière du soleil en énergie chimique.",
    },
    # 9
    {
        "en": "DNA is a double-helix molecule that carries genetic information in living organisms.",
        "es": "El ADN es una molécula en doble hélice que contiene la información genética de los organismos vivos.",
        "fr": "L'ADN est une molécule en double hélice qui porte l'information génétique des organismes vivants.",
    },
    # 10
    {
        "en": "The Mona Lisa is a famous portrait painted by Leonardo da Vinci.",
        "es": "La Mona Lisa es un famoso retrato pintado por Leonardo da Vinci.",
        "fr": "La Joconde est un célèbre portrait peint par Léonard de Vinci.",
    },
    # 11
    {
        "en": "A bicycle is a human-powered vehicle with two wheels and pedals.",
        "es": "Una bicicleta es un vehículo de propulsión humana con dos ruedas y pedales.",
        "fr": "Une bicyclette est un véhicule à propulsion humaine doté de deux roues et de pédales.",
    },
    # 12
    {
        "en": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
        "es": "Albert Einstein fue un físico teórico que desarrolló la teoría de la relatividad.",
        "fr": "Albert Einstein était un physicien théoricien qui a élaboré la théorie de la relativité.",
    },
    # 13
    {
        "en": "The Pythagorean theorem relates the sides of a right triangle through a squared equation.",
        "es": "El teorema de Pitágoras relaciona los lados de un triángulo rectángulo mediante una ecuación cuadrática.",
        "fr": "Le théorème de Pythagore relie les côtés d'un triangle rectangle par une équation au carré.",
    },
    # 14
    {
        "en": "World War II was a global conflict that lasted from 1939 to 1945.",
        "es": "La Segunda Guerra Mundial fue un conflicto global que duró de 1939 a 1945.",
        "fr": "La Seconde Guerre mondiale a été un conflit mondial qui a duré de 1939 à 1945.",
    },
    # 15
    {
        "en": "Penicillin is an antibiotic discovered by Alexander Fleming in 1928.",
        "es": "La penicilina es un antibiótico descubierto por Alexander Fleming en 1928.",
        "fr": "La pénicilline est un antibiotique découvert par Alexander Fleming en 1928.",
    },
    # 16
    {
        "en": "The Pyramids of Giza are ancient monumental tombs built in Egypt.",
        "es": "Las pirámides de Guiza son antiguas tumbas monumentales construidas en Egipto.",
        "fr": "Les pyramides de Gizeh sont d'anciennes tombes monumentales construites en Égypte.",
    },
    # 17
    {
        "en": "The Internet is a global network of computers communicating through TCP/IP protocols.",
        "es": "Internet es una red mundial de computadoras que se comunican mediante los protocolos TCP/IP.",
        "fr": "Internet est un réseau mondial d'ordinateurs qui communiquent via les protocoles TCP/IP.",
    },
    # 18
    {
        "en": "The Statue of Liberty is a colossal copper statue standing in New York Harbor.",
        "es": "La Estatua de la Libertad es una colosal estatua de cobre situada en el puerto de Nueva York.",
        "fr": "La Statue de la Liberté est une colossale statue en cuivre située dans le port de New York.",
    },
    # 19
    {
        "en": "Antarctica is the southernmost continent, almost entirely covered by ice.",
        "es": "La Antártida es el continente más austral, casi totalmente cubierto de hielo.",
        "fr": "L'Antarctique est le continent le plus austral, presque entièrement recouvert de glace.",
    },
    # 20
    {
        "en": "The lion is a large carnivorous feline often called the king of the jungle.",
        "es": "El león es un gran felino carnívoro a menudo llamado el rey de la selva.",
        "fr": "Le lion est un grand félin carnivore souvent surnommé le roi de la jungle.",
    },
    # 21
    {
        "en": "Sushi is a traditional Japanese dish made of vinegared rice and seafood.",
        "es": "El sushi es un plato tradicional japonés hecho con arroz avinagrado y mariscos.",
        "fr": "Le sushi est un plat traditionnel japonais à base de riz vinaigré et de fruits de mer.",
    },
    # 22
    {
        "en": "Yoga is a group of physical and mental practices originating in ancient India.",
        "es": "El yoga es un conjunto de prácticas físicas y mentales originarias de la antigua India.",
        "fr": "Le yoga est un ensemble de pratiques physiques et mentales originaires de l'Inde ancienne.",
    },
    # 23
    {
        "en": "The Solar System consists of the Sun and the eight planets that orbit it.",
        "es": "El sistema solar está compuesto por el Sol y los ocho planetas que lo orbitan.",
        "fr": "Le système solaire est composé du Soleil et des huit planètes qui gravitent autour de lui.",
    },
    # 24
    {
        "en": "Wolfgang Amadeus Mozart was a prolific classical composer of the eighteenth century.",
        "es": "Wolfgang Amadeus Mozart fue un prolífico compositor clásico del siglo XVIII.",
        "fr": "Wolfgang Amadeus Mozart était un compositeur classique prolifique du dix-huitième siècle.",
    },
    # 25
    {
        "en": "A volcano is a rupture in Earth's crust that allows molten lava to escape.",
        "es": "Un volcán es una ruptura en la corteza terrestre que permite la salida de lava fundida.",
        "fr": "Un volcan est une rupture dans la croûte terrestre qui laisse s'échapper de la lave en fusion.",
    },
    # 26
    {
        "en": "Chess is a strategy board game played on a board of sixty-four squares.",
        "es": "El ajedrez es un juego de estrategia de mesa que se juega en un tablero de sesenta y cuatro casillas.",
        "fr": "Les échecs sont un jeu de stratégie de plateau joué sur un damier de soixante-quatre cases.",
    },
    # 27
    {
        "en": "The FIFA World Cup is the most prestigious tournament in international football.",
        "es": "La Copa Mundial de la FIFA es el torneo más prestigioso del fútbol internacional.",
        "fr": "La Coupe du monde de la FIFA est le tournoi le plus prestigieux du football international.",
    },
    # 28
    {
        "en": "The Hubble Space Telescope is a NASA observatory orbiting Earth above the atmosphere.",
        "es": "El telescopio espacial Hubble es un observatorio de la NASA que orbita la Tierra por encima de la atmósfera.",
        "fr": "Le télescope spatial Hubble est un observatoire de la NASA en orbite autour de la Terre au-dessus de l'atmosphère.",
    },
    # 29
    {
        "en": "Tea is a popular aromatic beverage made by infusing leaves of the Camellia sinensis plant.",
        "es": "El té es una popular bebida aromática preparada infusionando hojas de la planta Camellia sinensis.",
        "fr": "Le thé est une boisson aromatique populaire préparée par infusion des feuilles de la plante Camellia sinensis.",
    },
]

assert len(concepts) == 30, f"Expected 30 concepts, got {len(concepts)}"

rows = []
for cid, triple in enumerate(concepts):
    for lang in ("en", "es", "fr"):
        rows.append({"concept_id": cid, "language": lang, "text": triple[lang]})

assert len(rows) == 90

out_path = "/home/user/myproject/corpus.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(rows)} rows to {out_path}")
PY

LABEL Author="generate-harbor-tasks"
LABEL Description="LanceDB cross-lingual search task using Cohere embed-multilingual-v3.0 on a baked 30x3 multilingual corpus."

CMD ["/bin/bash", "-c", "trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - always required apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep
#   - always required pip3 packages: pytest-json-ctrf
#   - target library: lancedb==0.25.3 with pyarrow==17.0.0, numpy==1.26.4, pandas==2.2.3
#   - Cohere SDK: cohere (latest 5.x) for embed-multilingual-v3.0 calls
#   - /home/user/myproject: candidate workspace
#   - /home/user/myproject/corpus.json: deterministic 30-concept x 3-language (EN/ES/FR) corpus = 90 rows
