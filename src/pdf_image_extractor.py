from pathlib import Path

import pymupdf


def extract_images_from_pdf(
    pdf_path: str,
    output_directory: str,
) -> list[dict]:
    """
    Extract embedded images from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        output_directory: Directory where extracted images will be saved.

    Returns:
        A list of dictionaries containing metadata about each image.
    """

    pdf_path = Path(pdf_path)
    output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    extracted_images = []

    for page_number, page in enumerate(document, start=1):
        images = page.get_images(full=True)

        for image_index, image in enumerate(images, start=1):
            xref = image[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            image_filename = (
                f"page_{page_number}_image_{image_index}.{image_extension}"
            )

            image_path = output_directory / image_filename

            image_path.write_bytes(image_bytes)

            extracted_images.append(
                {
                    "page_number": page_number,
                    "image_index": image_index,
                    "filename": image_filename,
                    "path": str(image_path),
                    "extension": image_extension,
                    "width": image_data["width"],
                    "height": image_data["height"],
                }
            )

    document.close()

    return extracted_images


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    output_directory = "data/extracted_images"

    images = extract_images_from_pdf(
        pdf_path,
        output_directory,
    )

    print("\nIMAGE EXTRACTION COMPLETE")
    print("=" * 60)

    if not images:
        print("No embedded images were found.")

    else:
        for image in images:
            print(
                f"Page {image['page_number']} | "
                f"Image {image['image_index']} | "
                f"{image['width']}x{image['height']} | "
                f"{image['path']}"
            )