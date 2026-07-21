from __future__ import annotations

from .models import Document


def demo_documents() -> list[Document]:
    """Return a deterministic fictional corpus for zero-credential demonstrations."""
    return [
        Document(
            document_id="demo:movie:signal-garden",
            title="The Signal Garden",
            media_kind="movie",
            year=2025,
            genres=("Science Fiction", "Drama"),
            source="moodlens-demo",
            text=(
                "A radio astronomer discovers that a repeating signal changes with the emotions "
                "of the people listening to it. She brings together a divided coastal town to "
                "decode the message before a private contractor turns it into a weapon."
            ),
        ),
        Document(
            document_id="demo:movie:paper-moons",
            title="Paper Moons",
            media_kind="movie",
            year=2024,
            genres=("Family", "Fantasy"),
            source="moodlens-demo",
            text=(
                "Two sisters build paper models of the moon while coping with their mother's "
                "absence. One model begins to predict the tides, leading them through a gentle "
                "story about grief, imagination, and reconciliation."
            ),
        ),
        Document(
            document_id="demo:movie:last-night-market",
            title="The Last Night Market",
            media_kind="movie",
            year=2026,
            genres=("Mystery", "Comedy"),
            source="moodlens-demo",
            text=(
                "A meticulous food inspector and an impulsive street magician investigate why "
                "an entire night market disappears before dawn. Their odd partnership turns a "
                "local mystery into a playful examination of memory and belonging."
            ),
        ),
        Document(
            document_id="demo:movie:winter-orchestra",
            title="The Winter Orchestra",
            media_kind="movie",
            year=2023,
            genres=("Music", "Drama"),
            source="moodlens-demo",
            text=(
                "After losing her hearing, a celebrated conductor returns to her mountain hometown "
                "and teaches an amateur orchestra to perform through vibration and visual cues. "
                "The story explores adaptation, pride, and community care."
            ),
        ),
        Document(
            document_id="demo:tv:borrowed-weather",
            title="Borrowed Weather",
            media_kind="tv",
            year=2025,
            genres=("Drama", "Speculative Fiction"),
            source="moodlens-demo",
            text=(
                "In a city where wealthy districts purchase perfect weather, a municipal engineer "
                "uncovers the cost paid by neighborhoods left with endless storms. Each episode "
                "follows residents organizing for environmental justice."
            ),
        ),
        Document(
            document_id="demo:tv:archive-of-small-things",
            title="The Archive of Small Things",
            media_kind="tv",
            year=2024,
            genres=("Mystery", "Drama"),
            source="moodlens-demo",
            text=(
                "An archivist catalogues ordinary objects donated after a citywide evacuation. "
                "Every object reveals an interconnected story, while clues hidden in the catalogue "
                "help her learn why the evacuation order was issued."
            ),
        ),
        Document(
            document_id="demo:tv:second-language",
            title="Second Language",
            media_kind="tv",
            year=2026,
            genres=("Comedy", "Romance"),
            source="moodlens-demo",
            text=(
                "A shy translator and a charismatic radio host accidentally exchange private "
                "voice notes. Their friendship grows through mistranslations and shared jokes, "
                "exploring vulnerability, migration, and the limits of perfect communication."
            ),
        ),
        Document(
            document_id="demo:tv:after-the-applause",
            title="After the Applause",
            media_kind="tv",
            year=2023,
            genres=("Drama",),
            source="moodlens-demo",
            text=(
                "Former child performers meet years after their television show ended and confront "
                "the adults who managed their careers. The ensemble drama examines fame, consent, "
                "nostalgia, and rebuilding an identity outside public attention."
            ),
        ),
    ]

