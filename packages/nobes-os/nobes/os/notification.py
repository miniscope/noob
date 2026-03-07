import asyncio
import sys
from typing import Annotated as A
from typing import Any
from typing import Literal as L

from desktop_notifier import Button, DesktopNotifier

from noob import Name
from noob.event import MetaSignal


async def notification(
    title: str,
    message: str,
    button_title: str | None = None,
    button_value: Any | None = None,
    app_name: str = "noob",
) -> tuple[A[bool | L[MetaSignal.NoEvent], Name("clicked")], A[Any, Name("value")]]:
    if sys.platform == "darwin":
        raise NotImplementedError("Mac makes it very hard to do notifications! PRs welcome!")
    else:
        impl = _win_linux_notification

    return await impl(
        title=title,
        message=message,
        button_title=button_title,
        button_value=button_value,
        app_name=app_name,
    )


async def _win_linux_notification(
    title: str,
    message: str,
    button_title: str | None = None,
    button_value: Any | None = None,
    app_name: str = "noob",
) -> tuple[A[bool | L[MetaSignal.NoEvent], Name("clicked")], A[Any, Name("value")]]:
    title = str(title)
    message = str(message)
    button_title = str(button_title) if button_title else None

    notifier = DesktopNotifier(app_name=app_name)
    clicked = MetaSignal.NoEvent
    return_value = MetaSignal.NoEvent
    event = asyncio.Event()
    event.clear()

    def on_click(**kwargs: Any) -> None:
        nonlocal clicked, event
        if button_title is None:
            clicked = True
        event.set()

    def on_dismiss(**kwargs: Any) -> None:
        nonlocal clicked, event
        clicked = MetaSignal.NoEvent
        event.set()

    def on_button(**kwargs: Any) -> None:
        nonlocal clicked, event, return_value, button_value
        return_value = button_value
        clicked = True
        event.set()

    buttons = [Button(title=button_title, on_pressed=on_button)] if button_title is not None else []

    await notifier.send(
        title=title, message=message, buttons=buttons, on_clicked=on_click, on_dismissed=on_dismiss
    )
    await event.wait()
    return clicked, return_value
