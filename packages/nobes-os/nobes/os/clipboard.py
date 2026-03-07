import clipboard


def set_clipboard(value: str) -> None:
    clipboard.copy(value)


def get_clipboard() -> str:
    return clipboard.paste()
