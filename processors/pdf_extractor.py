"""
PDF text extraction for AWS High Court judgments.

Uses PyPDF2 to extract text from PDF files, handling common issues
with scanned documents and encoding problems.
"""

import tarfile
import tempfile
from pathlib import Path

import PyPDF2

from utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string, or None if extraction failed.
    """
    try:
        pages_content = []
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        pages_content.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num} from {pdf_path}: {e}")
                    continue

        if not pages_content:
            logger.warning(f"No text extracted from {pdf_path}")
            return None

        # Join pages with double newline for separation
        return "\n\n".join(pages_content)

    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PDF read error for {pdf_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting text from {pdf_path}: {e}")
        return None


def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str = "unknown") -> str | None:
    """Extract text from PDF bytes (for in-memory processing).

    Args:
        pdf_bytes: PDF file content as bytes.
        filename: Filename for logging purposes.

    Returns:
        Extracted text as a single string, or None if extraction failed.
    """
    try:
        import io

        pages_content = []
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    pages_content.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} from {filename}: {e}")
                continue

        if not pages_content:
            logger.warning(f"No text extracted from {filename}")
            return None

        return "\n\n".join(pages_content)

    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PDF read error for {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting text from {filename}: {e}")
        return None


class TarPDFExtractor:
    """Extract and process PDFs from tar archives without fully extracting."""

    def __init__(self, tar_path: Path):
        """Initialize with path to tar file.

        Args:
            tar_path: Path to the tar archive containing PDFs.
        """
        self.tar_path = tar_path
        self._tar_file: tarfile.TarFile | None = None
        self._pdf_members: list[tarfile.TarInfo] | None = None

    def __enter__(self):
        """Open tar file for reading."""
        self._tar_file = tarfile.open(self.tar_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close tar file."""
        if self._tar_file:
            self._tar_file.close()
            self._tar_file = None

    def list_pdfs(self) -> list[str]:
        """List all PDF files in the tar archive.

        Returns:
            List of PDF filenames in the archive.
        """
        if not self._tar_file:
            raise RuntimeError("TarPDFExtractor must be used as context manager")

        if self._pdf_members is None:
            self._pdf_members = [
                m for m in self._tar_file.getmembers() if m.name.lower().endswith(".pdf")
            ]
        return [m.name for m in self._pdf_members]

    def get_cnr_from_filename(self, filename: str) -> str | None:
        """Extract CNR from PDF filename.

        Filename format: HCBM030079862025_1_2025-04-07.pdf
        CNR is the first part: HCBM030079862025

        Args:
            filename: PDF filename.

        Returns:
            CNR string or None if parsing failed.
        """
        try:
            # Get just the filename without path
            name = Path(filename).name
            # Split by underscore and take first part
            cnr = name.split("_")[0]
            return cnr
        except Exception:
            return None

    def extract_text(self, pdf_filename: str) -> str | None:
        """Extract text from a specific PDF in the tar archive.

        Args:
            pdf_filename: Name of the PDF file in the archive.

        Returns:
            Extracted text or None if extraction failed.
        """
        if not self._tar_file:
            raise RuntimeError("TarPDFExtractor must be used as context manager")

        try:
            member = self._tar_file.getmember(pdf_filename)
            file_obj = self._tar_file.extractfile(member)
            if file_obj is None:
                logger.warning(f"Could not extract {pdf_filename} from tar")
                return None

            pdf_bytes = file_obj.read()
            return extract_text_from_pdf_bytes(pdf_bytes, pdf_filename)

        except KeyError:
            logger.error(f"PDF {pdf_filename} not found in tar archive")
            return None
        except Exception as e:
            logger.error(f"Error extracting {pdf_filename} from tar: {e}")
            return None

    def iter_pdfs(self) -> "TarPDFIterator":
        """Iterate over all PDFs, yielding (cnr, text) tuples.

        Yields:
            Tuple of (cnr, extracted_text) for each PDF.
        """
        if not self._tar_file:
            raise RuntimeError("TarPDFExtractor must be used as context manager")

        if self._pdf_members is None:
            self._pdf_members = [
                m for m in self._tar_file.getmembers() if m.name.lower().endswith(".pdf")
            ]

        for member in self._pdf_members:
            cnr = self.get_cnr_from_filename(member.name)
            if not cnr:
                logger.warning(f"Could not parse CNR from {member.name}")
                continue

            try:
                file_obj = self._tar_file.extractfile(member)
                if file_obj is None:
                    continue

                pdf_bytes = file_obj.read()
                text = extract_text_from_pdf_bytes(pdf_bytes, member.name)
                yield cnr, text

            except Exception as e:
                logger.error(f"Error processing {member.name}: {e}")
                yield cnr, None


class TarPDFIterator:
    """Iterator for PDFs in a tar archive."""

    def __init__(self, extractor: TarPDFExtractor):
        self.extractor = extractor


def find_tar_files(base_path: Path, court_code: str | None = None, bench: str | None = None) -> list[Path]:
    """Find all tar files matching the given filters.

    Args:
        base_path: Base directory containing tar files.
        court_code: Filter by court code (e.g., "7_26").
        bench: Filter by bench (e.g., "dhcdb").

    Returns:
        List of paths to tar files.
    """
    tar_files = []

    # Build search pattern
    if court_code and bench:
        pattern = f"**/court={court_code}/bench={bench}/*.tar"
    elif court_code:
        pattern = f"**/court={court_code}/**/*.tar"
    elif bench:
        pattern = f"**/bench={bench}/*.tar"
    else:
        pattern = "**/*.tar"

    for tar_path in base_path.glob(pattern):
        # Skip part-*.tar files, only want data.tar or pdfs.tar
        if tar_path.name.startswith("part-"):
            continue
        tar_files.append(tar_path)

    return sorted(tar_files)


def get_tar_pdf_count(tar_path: Path) -> int:
    """Get count of PDFs in a tar file without fully reading it.

    Args:
        tar_path: Path to tar file.

    Returns:
        Number of PDF files in the archive.
    """
    try:
        with tarfile.open(tar_path, "r") as tf:
            return sum(1 for m in tf.getmembers() if m.name.lower().endswith(".pdf"))
    except Exception as e:
        logger.error(f"Error counting PDFs in {tar_path}: {e}")
        return 0
