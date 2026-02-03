import argparse
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from pysi.wind import Wind

from .viewer import make_document


def main():
    parser = argparse.ArgumentParser(description="Launch Wind Spectrum Viewer")
    parser.add_argument("root", help="Wind simulation root filename")
    parser.add_argument("--directory", default=".", help="Simulation directory")
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args()

    wind = Wind(root=args.root, directory=args.directory)

    def bkapp(doc):
        make_document(doc, wind)

    server = Server({"/": bkapp}, io_loop=IOLoop.current(), port=args.port)
    server.start()

    print(f"Opening Wind Spectrum Viewer at http://localhost:{args.port}/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
