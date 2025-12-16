from sphinx.util.docutils import SphinxDirective
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata
from docutils import nodes
from noob.tube import TubeSpecification
import re
import uuid

SCRIPT_TEMPLATE = """


window.addEventListener('load', () => {{
  let {tube_id}_spec = {tube_spec};
  window.renderPipeline("#tube-plot-{tube_id}", {tube_id}_spec);
}});
"""


class ScriptNode(nodes.TextElement): ...


def visit_script_html(self, node: ScriptNode):
    self.body.append(self.starttag(node, "script"))
    self.body.append(node.rawsource)


def depart_script_html(self, node: ScriptNode):
    self.body.append("</script>")


class NoobTubePlot(SphinxDirective):
    """Noob plot directive, renders a reactflow element for the given tube"""

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        spec = TubeSpecification.from_id(self.arguments[0])
        tube_id_esc = re.sub(r"[^a-zA-Z0-9]", "_", self.arguments[0] + "_" + str(uuid.uuid4()))
        container = nodes.container(classes=["noob-tube-container"])
        container["data-plot-for"] = f"tube-container-{tube_id_esc}"
        section = nodes.container(ids=[f"tube-plot-{tube_id_esc}"], classes=["noob-tube-plot"])
        container += section
        container += ScriptNode(
            SCRIPT_TEMPLATE.format(tube_id=tube_id_esc, tube_spec=spec.model_dump_json())
        )

        return [container]


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive("noob-tube", NoobTubePlot)
    app.add_node(ScriptNode, html=(visit_script_html, depart_script_html))
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
