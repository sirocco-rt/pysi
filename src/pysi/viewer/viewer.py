import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import (
    ColumnDataSource, TapTool, HoverTool,
    LinearColorMapper, ColorBar, Select
)


def make_document(doc, wind):
    xcen = wind["xcen"]
    zcen = wind["zcen"]
    logne = np.log10(wind["ne"])
    inwind = wind["inwind"]

    mask = (inwind == 0)

    xs = xcen[mask]
    zs = zcen[mask]
    ne_vals = logne[mask]

    i_idx, j_idx = np.where(mask)

    source = ColumnDataSource(dict(x=xs, z=zs, ne=ne_vals, i=i_idx, j=j_idx))

    color_mapper = LinearColorMapper(
        palette="Viridis256",
        low=float(ne_vals.min()),
        high=float(ne_vals.max())
    )

    p = figure(width=500, height=500, x_axis_type="log", y_axis_type="log", title="Wind Density")

    renderer = p.circle(
        "x", "z", source=source, size=7,
        fill_color={'field': 'ne', 'transform': color_mapper},
        line_color="black", line_width=0.1
    )

    p.add_tools(TapTool(renderers=[renderer]))
    p.add_tools(HoverTool(tooltips=[("i", "@i"), ("j", "@j"), ("log ne", "@ne")]))
    p.add_layout(ColorBar(color_mapper=color_mapper), 'right')

    # ---------------- Spectrum plot ----------------
    spec_source = ColumnDataSource(dict(freq=[], flux=[], mfreq=[], mflux=[]))

    q = figure(width=500, height=500, x_axis_type="log", y_axis_type="log", title="Spectrum")
    q.line("freq", "flux", source=spec_source, line_width=2)
    q.line("mfreq", "mflux", source=spec_source, line_width=2, color='#d62728')

    def update_spectrum(attr, old, new):
        if not new:
            return
        idx = new[0]
        i = i_idx[idx]
        j = j_idx[idx]

        spec_source.data = dict(
            freq=wind["spec_freq"][i, j],
            flux=wind["spec_flux"][i, j],
            mfreq=wind["model_freq"][i, j],
            mflux=wind["model_flux"][i, j]
        )

    source.selected.on_change("indices", update_spectrum)

    # ---------------- Physical properties panel ----------------
    select1 = Select(title="Property:", value="ne", options=[
        "rho", "ne", "t_e", "t_r", "ip", "xi", "inwind", "converge",
        "v_x", "v_y", "v_z", "vol", "ntot", "nrad", "nioniz",
        "nscat_es", "nscat_res", "nscat_ff", "nscat_bf",
        "v_l", "v_rot", "v_r"
    ])

    pcolor1_source = ColumnDataSource(dict(image=[wind["ne"].T]))
    color_mapper1 = LinearColorMapper(palette="Viridis256")

    r1 = figure(width=500, height=500, title="Physical Properties")
    r1.image(image="image", x=0, y=0, dw=1, dh=1,
             source=pcolor1_source, color_mapper=color_mapper1)
    r1.add_layout(ColorBar(color_mapper=color_mapper1), 'right')

    def update_pcolor1(attr, old, new):
        data = wind[select1.value]
        color_mapper1.low = float(np.nanmin(data))
        color_mapper1.high = float(np.nanmax(data))
        pcolor1_source.data = dict(image=[data.T])

    select1.on_change("value", update_pcolor1)

    # ---------------- Ion fractions panel ----------------
    select2 = Select(title="Fraction:", value="H_i01_frac", options=wind.ions_read_in)

    pcolor2_source = ColumnDataSource(dict(image=[wind["H_i01_frac"].T]))
    color_mapper2 = LinearColorMapper(palette="Viridis256")

    r2 = figure(width=500, height=500, title="Ionization Fractions")
    r2.image(image="image", x=0, y=0, dw=1, dh=1,
             source=pcolor2_source, color_mapper=color_mapper2)
    r2.add_layout(ColorBar(color_mapper=color_mapper2), 'right')

    def update_pcolor2(attr, old, new):
        data = wind[select2.value]
        color_mapper2.low = float(np.nanmin(data))
        color_mapper2.high = float(np.nanmax(data))
        pcolor2_source.data = dict(image=[data.T])

    select2.on_change("value", update_pcolor2)

    # ---------------- Layout ----------------
    doc.add_root(column(row(p, q), row(column(select1, r1), column(select2, r2))))
    doc.title = "Wind Spectrum Viewer"
