FROM ubuntu:24.04

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep locales && \
    locale-gen en_US.UTF-8 && \
    pip3 install --break-system-packages pytest-json-ctrf \
        lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 \
        tantivy==0.22.0 'jieba>=0.42' && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=UTF-8

RUN mkdir -p /home/user && chmod 0755 /home/user
RUN mkdir -p /home/user/myproject

RUN mkdir -p /opt/seed && cat >/opt/seed/seed.py <<'PYEOF'
import os
import sys

import lancedb
import pyarrow as pa

run_id = os.environ.get("ZEALT_RUN_ID")
if not run_id:
    print("ZEALT_RUN_ID not set; skipping seed.", file=sys.stderr)
    sys.exit(0)

db_dir = "/home/user/myproject/lancedb_data"
os.makedirs(db_dir, exist_ok=True)
db = lancedb.connect(db_dir)

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("content", pa.string()),
])

# English corpus.
#   id=1 is the rigged top-1 result for "running shoes" — it contains both
#   "running" and "shoes" multiple times in a short doc.
#   ids 2..5 each contain one of the stems (run / shoe) exactly once.
#   ids 6..40 contain neither "run" nor "shoe" stems, and intentionally
#   sprinkle the word "the" so queries containing common words still return
#   non-empty BM25 result sets when stop-word removal is disabled.
EN_DOCS = [
    "I love running shoes. My running shoes are perfect for jogging every morning. Running with these shoes feels amazing.",
    "She just bought new shoes for the upcoming summer travel adventure.",
    "He started running last year and has never looked back since.",
    "These shoes have excellent support for the long walks downtown.",
    "We went running together in the park yesterday morning before breakfast.",
    "The chef prepared a delicious pasta dish with the fresh tomato sauce.",
    "Music brings joy to many people around the world every single day.",
    "Painting landscapes is a relaxing hobby for the dedicated artist.",
    "She enjoys reading mystery novels by the fireplace at night.",
    "The mountain hike offered stunning views of the valley below the cliff.",
    "Photography captures moments that words sometimes cannot describe well.",
    "Coffee in the morning helps me focus on the project ahead today.",
    "Gardening teaches patience and the value of careful planning over time.",
    "Yoga every evening keeps the mind calm and the body flexible and fit.",
    "Travel broadens the perspective and enriches the soul over the years.",
    "Cooking a hearty soup on a cold day brings true comfort to the table.",
    "Astronomy fascinates many because the night sky holds endless wonder.",
    "Cycling along the river is a wonderful way to spend the afternoon.",
    "Pottery lessons are popular with people who enjoy hands-on art making.",
    "Writing fiction requires imagination and a love of the spoken words.",
    "Birdwatching in the forest is a peaceful activity for the soul.",
    "Carpentry skills are useful for building beautiful furniture at the home.",
    "The library has a wonderful collection of the historic manuscripts.",
    "Knitting sweaters during winter is a relaxing pastime for the family.",
    "Hiking the coastal trail provides breathtaking views of the ocean.",
    "Pizza night is a family tradition every Friday evening for the kids.",
    "The orchestra performed beautifully at the concert hall last night.",
    "Sailing across the lake at sunset is a magical experience for all.",
    "The aquarium showcases tropical fish from the many distant oceans.",
    "Volunteering at the food bank fills my heart with the deepest gratitude.",
    "Stargazing on clear nights reveals the beauty of the cosmos above.",
    "Painting the kitchen walls a soft yellow brightened the whole space.",
    "The wedding ceremony was held in a charming countryside garden setting.",
    "Skiing in the alpine resort attracted many enthusiastic visitors today.",
    "Cheese tasting is a delightful experience for any gourmet fan of food.",
    "The historic castle attracts thousands of tourists each summer season.",
    "Origami folding is an ancient art form practiced worldwide every day.",
    "The botanical garden features rare flowers from the tropical regions.",
    "Whale watching offers an unforgettable view of these majestic creatures.",
    "Beekeeping requires patience and care for the entire hive over time.",
]

# German corpus.
#   id=1 contains multiple forms of "laufen" (laufen / Laufen / laufe).
#   ids 2..5 each contain a single inflected form that stems to "lauf".
#   ids 6..40 contain NO words with the "lauf" substring at all.
DE_DOCS = [
    "Ich liebe das Laufen. Jeden Morgen gehe ich laufen, denn Laufen ist sehr gesund und macht mir wirklich Spaß.",
    "Wir laufen oft am Wochenende im Park zusammen mit unseren Freunden.",
    "Marathon laufen ist eine große sportliche Herausforderung für viele Menschen.",
    "Sie geht regelmäßig laufen, um fit und gesund zu bleiben.",
    "Beim Laufen am Strand höre ich gerne klassische Musik und entspanne mich.",
    "Die Köchin bereitet eine leckere Pasta mit frischer Tomatensoße zu.",
    "Musik bringt vielen Menschen Freude und Inspiration in den Alltag.",
    "Landschaften zu malen ist ein entspannendes Hobby für viele Künstler.",
    "Sie liest gerne Krimis am Kamin in kalten Winternächten zu Hause.",
    "Die Bergwanderung bietet wunderschöne Ausblicke auf das ruhige Tal.",
    "Fotografie hält Momente fest, die sich kaum beschreiben lassen.",
    "Kaffee am Morgen hilft mir bei der täglichen Arbeit im Büro sehr.",
    "Gartenarbeit lehrt Geduld und Wertschätzung für die heimische Natur.",
    "Yoga am Abend hält den Geist ruhig und den Körper geschmeidig zugleich.",
    "Reisen erweitert den Horizont und bereichert die Seele auf vielen Wegen.",
    "Eine heiße Suppe an einem kalten Tag spendet Wärme und Trost.",
    "Astronomie fasziniert viele Menschen wegen der Weite des Universums.",
    "Radfahren am Fluss ist eine schöne Beschäftigung am späten Nachmittag.",
    "Töpferkurse sind beliebt bei Menschen, die gerne handwerklich arbeiten.",
    "Romane schreiben verlangt Fantasie, Geduld und Hingabe an die Sprache.",
    "Vogelbeobachtung im Wald beruhigt die Seele und stärkt die Sinne.",
    "Tischlerei ermöglicht den Bau schöner Möbel zu Hause mit eigenen Händen.",
    "Die Bibliothek hat eine wunderbare Sammlung historischer alter Bücher.",
    "Stricken im Winter ist eine entspannende Beschäftigung für viele Menschen.",
    "Wandern entlang der Küste bietet atemberaubende Ausblicke auf das Meer.",
    "Pizzaabend ist bei uns jeden Freitag eine schöne familiäre Tradition.",
    "Das Orchester spielte gestern Abend wunderbar im großen Konzerthaus.",
    "Segeln über den See bei Sonnenuntergang ist ein magisches Erlebnis.",
    "Das Aquarium zeigt tropische Fische aus den fernen warmen Meeren.",
    "Freiwilligenarbeit im Tafelladen erfüllt mein Herz mit echter Dankbarkeit.",
    "Sternenbeobachtung in klaren Nächten zeigt die Schönheit des Kosmos.",
    "Die Küche in einem warmen Gelbton zu streichen, hellt den Raum spürbar auf.",
    "Die Hochzeit fand in einem charmanten Landgarten mit vielen Gästen statt.",
    "Skifahren im Alpenresort zog dieses Jahr viele begeisterte Besucher an.",
    "Käseverkostung ist ein wahrer Genuss für jeden ernsthaften Feinschmecker.",
    "Das historische Schloss zieht jedes Jahr Tausende Touristen aus aller Welt an.",
    "Origami ist eine alte Kunstform, die weltweit von vielen Menschen geübt wird.",
    "Der botanische Garten beherbergt seltene Blumen aus tropischen Gebieten.",
    "Walbeobachtung bietet einen unvergesslichen Blick auf diese großen Tiere.",
    "Imkerei erfordert Geduld und Pflege für den gesamten Bienenstock im Garten.",
]

# Chinese corpus (Simplified).
#   id=1 contains 跑步 multiple times in a short doc — guaranteed BM25 top-1.
#   ids 2..5 each contain 跑步 exactly once in a longer doc.
#   ids 6..40 do NOT contain the bigram 跑步 anywhere.
ZH_DOCS = [
    "我每天早上去公园跑步。跑步让我精力充沛，跑步真的非常棒。",
    "他每周末和朋友一起去湖边跑步，这是一项很好的运动。",
    "她每天清晨在江边跑步，已经坚持了好几年了。",
    "我们公司组织员工一起跑步，增强团队凝聚力和士气。",
    "在海边跑步的感觉非常美妙，可以呼吸新鲜的海风。",
    "厨师准备了一道美味的意大利面，配上新鲜的番茄酱。",
    "音乐给世界各地的人们带来欢乐和源源不断的灵感。",
    "画风景画是艺术家放松心情的一种很好的爱好。",
    "她喜欢在寒冷的夜晚靠着壁炉读侦探小说度过时光。",
    "山间徒步可以欣赏到壮丽的山谷美景和清新空气。",
    "摄影记录下许多无法用语言完整描述的瞬间。",
    "早晨的一杯咖啡让我专注于一天的工作和学习。",
    "园艺教会人们耐心和对大自然的尊重与热爱。",
    "每天晚上做瑜伽可以让身心保持平静和柔软的状态。",
    "旅行可以开阔视野，丰富心灵，让人生更有意义。",
    "在寒冷的日子里，一碗热汤带来真正的温暖和满足。",
    "天文学吸引很多人，因为夜空充满了无尽的奇迹。",
    "沿着河岸骑自行车是度过下午时光的绝佳方式。",
    "陶艺课程深受喜欢动手创作的人们的喜爱和欢迎。",
    "写小说需要丰富的想象力和对文字的深深热爱。",
    "在森林里观鸟是一种十分平静而美好的活动。",
    "木工技能可以帮助在家里制作漂亮实用的家具。",
    "图书馆收藏着许多珍贵的历史手稿和古老资料。",
    "冬天编织毛衣是一种放松心情的传统消遣方式。",
    "沿海岸徒步可以看到令人惊叹的海景和落日余晖。",
    "披萨之夜是我们家每周五晚上的家庭聚餐传统。",
    "管弦乐团昨晚在音乐厅演奏得非常精彩动人。",
    "夕阳下扬帆远航是一种美妙而难忘的体验。",
    "水族馆展出来自遥远海洋的各种热带鱼类。",
    "在食物银行做志愿者让我心怀感激和满足感。",
    "晴朗的夜晚仰望星空，可以揭示宇宙的壮丽美丽。",
    "把厨房的墙刷成柔和的黄色，让整个空间变得明亮。",
    "婚礼在一座迷人的乡村花园里举行，宾客如云。",
    "高山滑雪度假村吸引了许多热情的国内外游客。",
    "奶酪品鉴对每位美食爱好者来说都是一种享受。",
    "古老的城堡每年吸引数千名游客来参观游览。",
    "折纸是一种世界各地都在练习的古老传统艺术。",
    "植物园里种植着来自热带地区的稀有花卉品种。",
    "观鲸提供了观察这些雄伟海洋生物的难忘视角。",
    "养蜂需要对整个蜂巢的耐心和细致的关爱呵护。",
]

assert len(EN_DOCS) == 40, len(EN_DOCS)
assert len(DE_DOCS) == 40, len(DE_DOCS)
assert len(ZH_DOCS) == 40, len(ZH_DOCS)

for lang, docs in [("en", EN_DOCS), ("de", DE_DOCS), ("zh", ZH_DOCS)]:
    name = f"docs_{lang}_{run_id}"
    rows = [{"id": i + 1, "content": text} for i, text in enumerate(docs)]
    db.create_table(name, rows, schema=schema, mode="overwrite")

print("Seeded 3 tables for run", run_id, flush=True)
# lancedb 0.25.3 occasionally SIGABRTs during interpreter teardown when a
# connection holds open table handles; bypass __del__ chains explicitly.
os._exit(0)
PYEOF

LABEL Author="generate-harbor-tasks"
LABEL Description="Multilingual FTS (en/de/zh) task: three 40-doc LanceDB tables seeded at container start with rigged BM25 top-1 docs for each language."

CMD ["/bin/bash", "-c", "python3 /opt/seed/seed.py; trap : TERM INT; sleep infinity & wait"]

# Container contents:
#   - apt packages: curl wget ca-certificates git python3 python3-pip python3-pytest bash ripgrep locales
#   - pip packages: pytest-json-ctrf lancedb==0.25.3 pyarrow==17.0.0 numpy==1.26.4 pandas==2.2.3 tantivy==0.22.0 jieba>=0.42
#   - /home/user/myproject: candidate project root (lancedb_data is populated at container start)
#   - /opt/seed/seed.py: entrypoint seed creating docs_en_${ZEALT_RUN_ID}, docs_de_${ZEALT_RUN_ID}, docs_zh_${ZEALT_RUN_ID}
