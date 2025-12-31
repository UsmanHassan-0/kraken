import dash_core_components as dcc
import dash_html_components as html

# isort: off
from maindash import web_interface

# isort: on

from variables import DECORRELATION_OPTIONS, DOA_METHODS, option


def get_dsp_config_card_layout():
    # Calulcate spacings
    ant_spacing_meter = web_interface.ant_spacing_meters
    en_doa_values = [1] if web_interface.module_signal_processor.en_DOA_estimation else []
    # -----------------------------
    #    DSP Confugartion Card
    # -----------------------------

    return html.Div(
        [
            html.H2("DoA Configuration", id="init_title_d"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Array Radius [m]:", id="label_ant_spacing_meter", className="field-label"),
                            dcc.Input(
                                id="ant_spacing_meter",
                                value=ant_spacing_meter,
                                type="number",
                                step=0.001,
                                min=0.001,
                                debounce=True,
                                className="field-body-textbox",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Wavelength Multiplier:", id="label_ant_spacing_wavelength", className="field-label"
                            ),
                            html.Div("1", id="body_ant_spacing_wavelength", className="field-body"),
                        ],
                        className="field",
                    ),
                ],
                id="antspacing",
                className="field",
            ),
            html.Div([html.Div("", id="ambiguity_warning", className="field", style={"color": "#f39c12"})]),
            # --> DoA estimation configuration checkboxes <--
            # Note: Individual checkboxes are created due to layout
            # considerations, correct if extist a better solution
            html.Div(
                [
                    html.Div("Enable DoA Estimation:", id="label_en_doa", className="field-label"),
                    dcc.Checklist(options=option, id="en_doa_check", className="field-body", value=en_doa_values),
                ],
                className="field",
            ),
            html.Div(
                [
                    html.Div("DoA Algorithm:", id="label_doa_method", className="field-label"),
                    dcc.Dropdown(
                        id="doa_method",
                        options=DOA_METHODS,
                        value=web_interface.module_signal_processor.DOA_algorithm,
                        style={"display": "inline-block"},
                        className="field-body",
                    ),
                ],
                className="field",
            ),
            html.Div(
                [
                    html.Div("Decorrelation:", id="label_decorrelation", className="field-label"),
                    dcc.Dropdown(
                        id="doa_decorrelation_method",
                        options=DECORRELATION_OPTIONS,
                        value=web_interface.module_signal_processor.DOA_decorrelation_method,
                        style={"display": "inline-block"},
                        className="field-body",
                    ),
                ],
                className="field",
            ),
            html.Div([html.Div("", id="uca_decorrelation_warning", className="field", style={"color": "#f39c12"})]),
            html.Div(
                [
                    html.Div("Array Offset:", id="label_array_offset", className="field-label"),
                    dcc.Input(
                        id="array_offset",
                        value=web_interface.module_signal_processor.array_offset,
                        type="number",
                        className="field-body-textbox",
                        debounce=True,
                    ),
                ],
                className="field",
            ),
            html.Div(
                [
                    html.Div(
                        "Expected number of RF sources:", id="label_expected_num_of_sources", className="field-label"
                    ),
                    dcc.Dropdown(
                        id="expected_num_of_sources",
                        options=[
                            {"label": "1", "value": 1},
                            {"label": "2", "value": 2},
                            {"label": "3", "value": 3},
                            {"label": "4", "value": 4},
                        ],
                        value=web_interface.module_signal_processor.DOA_expected_num_of_sources,
                        style={"display": "inline-block"},
                        className="field-body",
                    ),
                ],
                className="field",
            ),
        ],
        className="card",
    )
