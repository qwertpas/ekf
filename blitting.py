
import sys
import matplotlib.pyplot as plt
import numpy as np


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


if __name__ == "__main__":

    fig, ax = plt.subplots()
    def on_close(event):
        sys.exit()
    fig.canvas.mpl_connect('close_event', on_close)

    x = np.linspace(0, 100, 100)
    data = np.zeros_like(x)

    ax.set_ylim(-1, 1)

    (ln,) = ax.plot(x, data, 'o-', animated=True)

    bm = BlitManager(fig.canvas, [ln])
    # make sure our window is on the screen and drawn
    plt.show(block=False)
    plt.pause(.1)



    while True:
        # update the artists
        data = np.roll(data, -1)
        data[-1] = np.random.normal(loc=0.5, scale=0.25)
        ln.set_ydata(data)
        ln.set_xdata(x)

        # tell the blitting manager to do its thing
        bm.update()
        plt.pause(0.02)
