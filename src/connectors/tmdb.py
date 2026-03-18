"""
TMDBConnector — wraps the existing TMDB RAG pipeline and agent
into the ProbeConnector interface.

This doesn't change any existing code. It's a thin wrapper that
translates between ProbeConnector's domain-agnostic types and
the TMDB-specific pipeline internals.

Schema and corpus samples work even without the DB running —
useful for scenario generation on machines without PostgreSQL.
"""

import random

from loguru import logger

from src.connectors.base import (
    CorpusSchema,
    FieldInfo,
    ProbeConnector,
    ProbeResult,
    SourceDocument,
)

# known TMDB movies for offline sample generation — enough variety
# for the scenario generator to create good questions
_OFFLINE_SAMPLES = [
    SourceDocument(
        doc_id="27205", title="Inception",
        content=(
            "Title: Inception\nDirector: Christopher Nolan\n"
            "Cast: Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy, Ken Watanabe\n"
            "Genres: Action, Science Fiction, Adventure\nRelease Year: 2010\n"
            "Budget: $160,000,000 | Revenue: $835,532,764\nRuntime: 148 minutes\n"
            "Rating: 8.1/10 (14075 votes)\nKeywords: dream, subconscious, heist, mission, sleep\n"
            "Production: Warner Bros., Legendary Pictures, Syncopy\n"
            "Plot: A thief who steals corporate secrets through the use of dream-sharing "
            "technology is given the inverse task of planting an idea into the mind of a CEO."
        ),
        metadata={"movie_id": 27205, "director": "Christopher Nolan", "genres": ["Action", "Science Fiction", "Adventure"]},
    ),
    SourceDocument(
        doc_id="155", title="The Dark Knight",
        content=(
            "Title: The Dark Knight\nDirector: Christopher Nolan\n"
            "Cast: Christian Bale, Heath Ledger, Aaron Eckhart, Michael Caine, Gary Oldman\n"
            "Genres: Drama, Action, Crime, Thriller\nRelease Year: 2008\n"
            "Budget: $185,000,000 | Revenue: $1,004,558,444\nRuntime: 152 minutes\n"
            "Rating: 8.2/10 (12269 votes)\nKeywords: joker, batman, gotham city, moral dilemma\n"
            "Production: Warner Bros., Legendary Pictures, DC Comics\n"
            "Plot: When the menace known as the Joker wreaks havoc on Gotham, Batman must "
            "accept one of the greatest psychological and physical tests of his ability to fight injustice."
        ),
        metadata={"movie_id": 155, "director": "Christopher Nolan", "genres": ["Drama", "Action", "Crime", "Thriller"]},
    ),
    SourceDocument(
        doc_id="19995", title="Avatar",
        content=(
            "Title: Avatar\nDirector: James Cameron\n"
            "Cast: Sam Worthington, Zoe Saldana, Sigourney Weaver, Stephen Lang, Michelle Rodriguez\n"
            "Genres: Action, Adventure, Fantasy, Science Fiction\nRelease Year: 2009\n"
            "Budget: $237,000,000 | Revenue: $2,787,965,087\nRuntime: 162 minutes\n"
            "Rating: 7.2/10 (11800 votes)\nKeywords: alien, marine, pandora, na'vi, colonialism\n"
            "Production: Twentieth Century Fox, Lightstorm Entertainment\n"
            "Plot: In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora "
            "on a unique mission, but becomes torn between following orders and protecting an alien civilization."
        ),
        metadata={"movie_id": 19995, "director": "James Cameron", "genres": ["Action", "Adventure", "Fantasy", "Science Fiction"]},
    ),
    SourceDocument(
        doc_id="862", title="Toy Story",
        content=(
            "Title: Toy Story\nDirector: John Lasseter\n"
            "Cast: Tom Hanks, Tim Allen, Don Rickles, Jim Varney, Wallace Shawn\n"
            "Genres: Animation, Comedy, Family\nRelease Year: 1995\n"
            "Budget: $30,000,000 | Revenue: $373,554,033\nRuntime: 81 minutes\n"
            "Rating: 7.7/10 (5415 votes)\nKeywords: toy, jealousy, friendship, boy, rivalry\n"
            "Production: Pixar Animation Studios\n"
            "Plot: A cowboy doll is profoundly threatened and jealous when a new spaceman "
            "figure supplants him as top toy in a boy's room."
        ),
        metadata={"movie_id": 862, "director": "John Lasseter", "genres": ["Animation", "Comedy", "Family"]},
    ),
    SourceDocument(
        doc_id="550", title="Fight Club",
        content=(
            "Title: Fight Club\nDirector: David Fincher\n"
            "Cast: Brad Pitt, Edward Norton, Helena Bonham Carter, Meat Loaf, Jared Leto\n"
            "Genres: Drama\nRelease Year: 1999\n"
            "Budget: $63,000,000 | Revenue: $100,853,753\nRuntime: 139 minutes\n"
            "Rating: 8.3/10 (9678 votes)\nKeywords: dual identity, insomnia, anarchy, nihilism\n"
            "Production: Fox 2000 Pictures, Regency Enterprises, Linson Films\n"
            "Plot: An insomniac office worker and a devil-may-care soapmaker form an underground "
            "fight club that evolves into something much, much more."
        ),
        metadata={"movie_id": 550, "director": "David Fincher", "genres": ["Drama"]},
    ),
    SourceDocument(
        doc_id="680", title="Pulp Fiction",
        content=(
            "Title: Pulp Fiction\nDirector: Quentin Tarantino\n"
            "Cast: John Travolta, Uma Thurman, Samuel L. Jackson, Bruce Willis, Tim Roth\n"
            "Genres: Thriller, Crime\nRelease Year: 1994\n"
            "Budget: $8,000,000 | Revenue: $213,928,762\nRuntime: 154 minutes\n"
            "Rating: 8.3/10 (8670 votes)\nKeywords: nonlinear timeline, hitman, robbery, drugs\n"
            "Production: Miramax, A Band Apart, Jersey Films\n"
            "Plot: The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of "
            "diner bandits intertwine in four tales of violence and redemption."
        ),
        metadata={"movie_id": 680, "director": "Quentin Tarantino", "genres": ["Thriller", "Crime"]},
    ),
    SourceDocument(
        doc_id="278", title="The Shawshank Redemption",
        content=(
            "Title: The Shawshank Redemption\nDirector: Frank Darabont\n"
            "Cast: Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler, Clancy Brown\n"
            "Genres: Drama, Crime\nRelease Year: 1994\n"
            "Budget: $25,000,000 | Revenue: $28,341,469\nRuntime: 142 minutes\n"
            "Rating: 8.5/10 (8358 votes)\nKeywords: prison, wrongful imprisonment, hope, escape\n"
            "Production: Castle Rock Entertainment\n"
            "Plot: Two imprisoned men bond over a number of years, finding solace and eventual "
            "redemption through acts of common decency."
        ),
        metadata={"movie_id": 278, "director": "Frank Darabont", "genres": ["Drama", "Crime"]},
    ),
    SourceDocument(
        doc_id="597", title="Titanic",
        content=(
            "Title: Titanic\nDirector: James Cameron\n"
            "Cast: Leonardo DiCaprio, Kate Winslet, Billy Zane, Kathy Bates, Bill Paxton\n"
            "Genres: Drama, Romance\nRelease Year: 1997\n"
            "Budget: $200,000,000 | Revenue: $1,845,034,188\nRuntime: 194 minutes\n"
            "Rating: 7.5/10 (7562 votes)\nKeywords: ship, love, iceberg, disaster, class difference\n"
            "Production: Paramount Pictures, Twentieth Century Fox, Lightstorm Entertainment\n"
            "Plot: A seventeen-year-old aristocrat falls in love with a kind but poor artist "
            "aboard the luxurious, ill-fated R.M.S. Titanic."
        ),
        metadata={"movie_id": 597, "director": "James Cameron", "genres": ["Drama", "Romance"]},
    ),
    SourceDocument(
        doc_id="603", title="The Matrix",
        content=(
            "Title: The Matrix\nDirector: Lana Wachowski\n"
            "Cast: Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving\n"
            "Genres: Action, Science Fiction\nRelease Year: 1999\n"
            "Budget: $63,000,000 | Revenue: $463,517,383\nRuntime: 136 minutes\n"
            "Rating: 8.1/10 (9079 votes)\nKeywords: simulated reality, hacker, chosen one, martial arts\n"
            "Production: Warner Bros., Village Roadshow, Silver Pictures\n"
            "Plot: A computer hacker learns from mysterious rebels about the true nature of "
            "his reality and his role in the war against its controllers."
        ),
        metadata={"movie_id": 603, "director": "Lana Wachowski", "genres": ["Action", "Science Fiction"]},
    ),
    SourceDocument(
        doc_id="120", title="The Lord of the Rings: The Fellowship of the Ring",
        content=(
            "Title: The Lord of the Rings: The Fellowship of the Ring\nDirector: Peter Jackson\n"
            "Cast: Elijah Wood, Ian McKellen, Orlando Bloom, Viggo Mortensen, Sean Astin\n"
            "Genres: Adventure, Fantasy, Action\nRelease Year: 2001\n"
            "Budget: $93,000,000 | Revenue: $871,368,364\nRuntime: 178 minutes\n"
            "Rating: 8.0/10 (8892 votes)\nKeywords: ring, quest, hobbit, middle earth, fellowship\n"
            "Production: New Line Cinema, WingNut Films\n"
            "Plot: A meek Hobbit from the Shire and eight companions set out on a journey "
            "to destroy the powerful One Ring and save Middle-earth."
        ),
        metadata={"movie_id": 120, "director": "Peter Jackson", "genres": ["Adventure", "Fantasy", "Action"]},
    ),
]


class TMDBConnector(ProbeConnector):
    """Connector for the built-in TMDB 5000 movie RAG + agent system."""

    def ask(self, question: str, mode: str = "rag") -> ProbeResult:
        if mode == "agent":
            return self._ask_agent(question)
        return self._ask_rag(question)

    def _ask_rag(self, question: str) -> ProbeResult:
        from src.rag.pipeline import ask

        result = ask(question)

        source_docs = [
            SourceDocument(
                doc_id=str(s.movie_id),
                title=s.title,
                content=s.chunk_text,
                metadata={"movie_id": s.movie_id},
                similarity=s.similarity,
            )
            for s in result.sources
        ]

        return ProbeResult(
            question=result.question,
            answer=result.answer,
            source_documents=source_docs,
            context_texts=result.context_texts,
            latency_ms=result.latency_ms,
            token_usage=result.generation.usage,
            model=result.generation.model,
        )

    def _ask_agent(self, question: str) -> ProbeResult:
        from src.agent.graph import run_agent

        result = run_agent(question)

        # agent doesn't do vector retrieval, but its tool outputs
        # serve as context for faithfulness/grounding checks
        context_texts = []
        for tc in result.trace.tool_calls:
            if tc.tool_output:
                ctx = str(tc.tool_output)
                if len(ctx) > 3000:
                    ctx = ctx[:3000] + "..."
                context_texts.append(ctx)

        return ProbeResult(
            question=result.question,
            answer=result.answer,
            source_documents=[],  # no vector retrieval in agent mode
            context_texts=context_texts,
            latency_ms=result.latency_ms,
            token_usage={
                "input_tokens": result.trace.total_input_tokens,
                "output_tokens": result.trace.total_output_tokens,
            },
            model="claude-sonnet-4-5-20250929 (agent)",
            agent_trace=result.trace,
        )

    def get_corpus_sample(self, n: int = 10) -> list[SourceDocument]:
        # try the database first; fall back to offline samples
        # so scenario generation works even without PostgreSQL
        try:
            return self._corpus_sample_from_db(n)
        except Exception:
            logger.info("DB unavailable — using offline TMDB samples")
            samples = list(_OFFLINE_SAMPLES)
            random.shuffle(samples)
            return samples[:n]

    def _corpus_sample_from_db(self, n: int) -> list[SourceDocument]:
        from sqlalchemy import func

        from src.database.connection import get_session
        from src.database.models import MovieChunk

        session = get_session()
        try:
            rows = session.query(MovieChunk).order_by(func.random()).limit(n).all()
            return [
                SourceDocument(
                    doc_id=str(row.movie_id),
                    title=row.title,
                    content=row.chunk_text,
                    metadata={"movie_id": row.movie_id},
                )
                for row in rows
            ]
        finally:
            session.close()

    def get_schema(self) -> CorpusSchema:
        # try to get the real count from DB; fall back to known dataset size
        total = 4803  # known TMDB 5000 dataset size after dedup
        try:
            from sqlalchemy import func

            from src.database.connection import get_session
            from src.database.models import MovieChunk

            session = get_session()
            try:
                total = session.query(func.count(MovieChunk.id)).scalar() or total
            finally:
                session.close()
        except Exception:
            logger.info("DB unavailable — using known TMDB dataset size")

        return CorpusSchema(
            domain_name="movies",
            description=(
                "TMDB 5000 movie dataset with plot overviews, cast, crew, "
                "financials, and ratings. Each movie is one document chunk "
                "combining structured metadata and unstructured plot text."
            ),
            fields=[
                FieldInfo(
                    name="title",
                    field_type="string",
                    description="Movie title",
                    sample_values=["Inception", "The Dark Knight", "Avatar"],
                ),
                FieldInfo(
                    name="director",
                    field_type="string",
                    description="Film director",
                    sample_values=["Christopher Nolan", "James Cameron", "Steven Spielberg"],
                ),
                FieldInfo(
                    name="cast",
                    field_type="list",
                    description="Top-billed actors",
                    sample_values=["Leonardo DiCaprio", "Tom Hardy", "Christian Bale"],
                ),
                FieldInfo(
                    name="genres",
                    field_type="list",
                    description="Genre tags",
                    sample_values=["Action", "Science Fiction", "Drama", "Comedy"],
                ),
                FieldInfo(
                    name="release_year",
                    field_type="number",
                    description="Year of release",
                    sample_values=["2010", "2008", "1994"],
                ),
                FieldInfo(
                    name="budget",
                    field_type="number",
                    description="Production budget in USD",
                    sample_values=["160000000", "185000000", "237000000"],
                ),
                FieldInfo(
                    name="revenue",
                    field_type="number",
                    description="Box office revenue in USD",
                    sample_values=["835532764", "1004558444", "2787965087"],
                ),
                FieldInfo(
                    name="rating",
                    field_type="number",
                    description="Average user rating out of 10",
                    sample_values=["8.1", "8.2", "7.2"],
                ),
                FieldInfo(
                    name="runtime",
                    field_type="number",
                    description="Runtime in minutes",
                    sample_values=["148", "152", "162"],
                ),
                FieldInfo(
                    name="keywords",
                    field_type="list",
                    description="Thematic keywords",
                    sample_values=["dream", "heist", "superhero", "revenge"],
                ),
                FieldInfo(
                    name="plot",
                    field_type="string",
                    description="Plot overview / synopsis",
                    sample_values=["A thief who steals corporate secrets through dream-sharing technology..."],
                ),
            ],
            total_documents=total,
        )
