class Axis:
    """Mixin providing common axis-related attributes."""

    frames_axis: str = "frame"
    """Name of the dimension representing time points."""

    spatial_axes: tuple[str, str] = ("height", "width")
    """Names of the dimensions representing 2-d spatial coordinates Default: (height, width)."""

    component_axis: str = "component"
    """Name of the dimension representing individual components."""

    id_coordinates: str = "id_"
    """Name of the coordinate used to identify individual components with unique IDs."""

    type_coordinates: str = "type_"
    """Name of the coordinate used to specify component types (e.g., neuron, background)."""

    frame_coordinates: str = "frame_"

    time_coordinates: str = "time_"
