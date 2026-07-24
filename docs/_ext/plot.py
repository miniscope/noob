import json
import re
import uuid
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from pydantic_core import PydanticSerializationError
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata

from noob.tube import TubeSpecification

logger = logging.getLogger(__name__)
EXTRA_DEPS = ["faker"]

SCRIPT_TEMPLATE = """


window.addEventListener('load', () => {{
  let {tube_id}_spec = {tube_spec};
  window.renderPipeline("#tube-plot-{tube_id}", {tube_id}_spec);
}});
"""

RUNNABLE_SCRIPT_TEMPLATE = """


window.addEventListener('load', () => {{
  let {tube_id}_spec = {tube_spec};
  let {tube_id}_options = {options};
  // resolve bundled wheel paths relative to the docs root, wherever this page is
  const root = document.documentElement.dataset.content_root ?? "";
  {tube_id}_options.wheels = {tube_id}_options.wheels.map(
    (wheel) => (/^https?:/.test(wheel) ? wheel : root + wheel)
  );
  window.renderRunnableTube("#tube-plot-{tube_id}", {tube_id}_spec, {tube_id}_options);
}});
"""


class ScriptNode(nodes.TextElement): ...


def visit_script_html(self, node: ScriptNode):
    self.body.append(self.starttag(node, "script"))
    self.body.append(node.rawsource)


def depart_script_html(self, node: ScriptNode):
    self.body.append("</script>")


class NoobTubePlot(SphinxDirective):
    """
    Noob plot directive, renders a reactflow element for the given tube.

    With the ``:runnable:`` flag, the tube gets runner controls backed by a
    lazily loaded pyodide runtime. Wheel URLs come from the
    ``noob_runner_wheels`` config value, or by scanning ``_static/wheels/``.
    """

    required_arguments = 1
    option_spec = {"runnable": directives.flag}

    def run(self) -> list[nodes.Node]:
        spec = TubeSpecification.from_id(self.arguments[0], context={"recursive": True})
        tube_id_esc = re.sub(r"[^a-zA-Z0-9]", "_", self.arguments[0] + "_" + str(uuid.uuid4()))
        container = nodes.container(classes=["noob-tube-container"])
        container["data-plot-for"] = f"tube-container-{tube_id_esc}"
        # runnable mounts hold controls + graph + event table and size
        # themselves; the fixed-height plot class would clip them
        mount_class = "noob-tube-runnable" if "runnable" in self.options else "noob-tube-plot"
        section = nodes.container(ids=[f"tube-plot-{tube_id_esc}"], classes=[mount_class])
        container += section

        # nodeinfo gives the JS side its signal/slot handles, but computing it
        # imports the node types - illustrative tubes with fictional types
        # render without it
        try:
            tube_spec = spec.model_dump_json()
        except PydanticSerializationError:
            tube_spec = spec.model_dump_json(exclude={"nodes": {"__all__": {"nodeinfo"}}})
        if "runnable" in self.options:
            script = RUNNABLE_SCRIPT_TEMPLATE.format(
                tube_id=tube_id_esc,
                tube_spec=tube_spec,
                options=json.dumps(self._runner_options()),
            )
        else:
            script = SCRIPT_TEMPLATE.format(tube_id=tube_id_esc, tube_spec=tube_spec)
        container += ScriptNode(script)

        return [container]

    def _runner_options(self) -> dict:
        config = self.env.app.config
        wheels = config.noob_runner_wheels
        if wheels is None:
            wheels = [
                f"_static/wheels/{wheel.name}"
                for wheel in sorted(Path(self.env.app.confdir, "_static", "wheels").glob("*.whl"))
            ]
        if not wheels:
            logger.warning(
                "%s is :runnable: but no wheels were found - "
                "set noob_runner_wheels or build wheels into _static/wheels/",
                self.arguments[0],
                location=self.get_location(),
            )
        options = {"wheels": wheels}
        if config.noob_runner_pyodide_url is not None:
            options["pyodideUrl"] = config.noob_runner_pyodide_url
        options['extra_deps'] = EXTRA_DEPS
        return options


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("noob-tube", NoobTubePlot)
    app.add_node(ScriptNode, html=(visit_script_html, depart_script_html))
    app.add_config_value("noob_runner_wheels", None, "html", types=[list])
    app.add_config_value("noob_runner_pyodide_url", None, "html", types=[str])
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
