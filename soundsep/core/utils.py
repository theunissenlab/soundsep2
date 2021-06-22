import numpy as np


def get_blocks_of_ones(vec: np.ndarray):
    """For a binary vector, return the indices surrounding contiguous blocks of ones

    Arguments
    ---------
    vec : np.ndarray
        A binary array of 1's and 0's (or True and False values)

    Example
    -------
    >>> vec = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    >>> get_blocks_of_ones(vec)
    array([[ 2, 5],
           [ 7, 8]])

    Returns
    -------
    breaks : np.ndarray (shape (N, 2))
        The start of stop indices of N continguous blocks of 1's in the data.
        The start point is inclusive and end point is non-inclusive, so can be
        used for slicing.
    """
    if not len(vec):
        return np.zeros((0, 2))

    breaks = np.where(vec[:-1] != vec[1:])[0] + 1
    if vec[0] == 1:
        breaks = np.concatenate([[0], breaks])
    if vec[-1] == 1:
        breaks = np.concatenate([breaks, [len(vec)]])
    return breaks.reshape((-1, 2))


class DuplicatedRingBuffer:
    """A ring buffer that doubles memory usage to provide fast reads and rolls

    In Soundsep the spectrogram often needs to be moved along the time axis in short
    distances. Each time this happens, the cache needs to adjust by rolling its data
    to the new start position, and the application then reads from the cache to fill
    in the image displayed. This implementation avoids needing to allocate data during
    rolls or reads at the cost of duplicating the data in a continuous block.

    The data for a buffer of size N is stored on disk as an array of size 2N, with a
    pointer in [0, N) that represents the zero point of the readable data block (labeled
    as ``^ptr``. The outside world sees the data marked by ``x``'s:

      ``[          N          ][          N          ]``
      ``xxxxxxxxxxxxxxxxxxxxxxx                       ``
      ``^ptr                                          ``

    This whenever data is written, it is duplicated in both halves:

      ``[    1234             ][    1234             ]``
      ``^ptr                                          ``

    When the data nees to be "rolled" by an offset n, the ptr is just moved by n, then
    moduloed N. The outside world sees the data marked by ``x``'s. Notice that since
    ``ptr`` is always < N, the data read is always in one continguous block and so
    can always be read as a numpy view (no data copying necessary).

      ``[    1234             ][    1234             ]``
      ``       xxxxxxxxxxxxxxxxxxxxxxx                ``
      ``       ^ptr                                   ``

    Arguments
    ---------
    from_array : np.ndarray
        The numpy array with shape and dtype that will be preserved. Internally, the
        DuplicatedRingBuffer will use twice that size.

    Examples
    --------
    >>> ring = DuplicatedRingBuffer(np.arange(100)[:, None])
    >>> ring[10:15]
    array([[10],
           [11],
           [12],
           [13],
           [14]])
    >>> ring.roll(10, fill=-1)
    >>> ring[10:15]
    array([[0],
           [1],
           [2],
           [3],
           [4]])
    >>> ring[:5]
    array([[-1],
           [-1],
           [-1],
           [-1],
           [-1]])
    """
    def __init__(self, from_array):
        self._n = from_array.shape[0]
        self._data = np.concatenate([from_array, from_array])
        self._ptr = 0

    def __len__(self):
        return self._n

    def __repr__(self):
        return "DuplicatedRingBuffer<{}>".format(self[:])

    def roll(self, offset, fill):
        """Roll the buffer in place by an offset

        Arguments
        ---------
        offset : int
            Number of samples to roll the data
        fill
            Value to fill in samples that rolled over the opposite end
        """
        # If we are trying to roll the entire array, just blank it all out
        # and reset
        if offset >= self._n:
            self._ptr = 0
            self[:] = fill
            return

        self._ptr += offset
        self._ptr %= self._n
        if offset > 0:
            self[-offset:] = fill
        elif offset < 0:
            self[:-offset] = fill

    @property
    def shape(self):
        return (self._n,) + self._data.shape[1:]

    @property
    def size(self):
        return self._data.size // 2

    def _convert_index(self, i):
        """Convert an external index into an internal index

        Arguments
        ---------
        i : int
            The external index you want to access

        Returns
        -------
        j : int
            The internal index you would use to read data at i

        Example
        -------
        >>> roll = DuplicatedRingBuffer(...)
        >>> roll._ptr
        0
        >>> roll._convert_index(10)
        10
        >>> roll.roll(15, ...)
        >>> roll._ptr
        15
        >>> roll._convert_index(10)
        25
        """
        return (self._ptr + i) % len(self)

    def _duplicated_index(self, i):
        """Get internal index to duplicate data at i at

        Arguments
        ---------
        i : int
            The internal index that you want to write at

        Returns
        -------
        j : int
            The internal index you want to duplicate writes at i at
        """
        N = len(self)
        return (i + N) % (2 * N)

    def _convert_slice(self, slice_):
        """Convert a slice into the internal slice coordinates to read/write at

        Arguments
        ---------
        slice_ : slice
            The external slice of data you want to access

        Returns
        -------
        internal_slice : slice
            The internal slice of data you would use to read data of slice_ at
        """
        N = len(self)
        start, stop, step = slice_.indices(N)
        start += self._ptr
        stop += self._ptr
        return slice(start, stop, step)

    def _duplicated_slice(self, slice_):
        """Convert an internal slice of data where you would write into the slices of data you would duplicate at

        Arguments
        ---------
        slice_ : slice
            The internal slice of data you want to write to

        Returns
        -------
        duplicated_slice : np.ndarray[int]
            The indices (using np.r_) of where the data should be duplicated
        """
        N = len(self)
        start, stop, step = slice_.indices(2 * N)
        wrapped_stop = stop % N
        wrapped_start = (start + step - N) % step
        return np.r_[
            slice(start + N, 2 * N, step),
            slice(wrapped_start, wrapped_start, step)
        ]

    def __getitem__(self, selectors):
        if isinstance(selectors, (int, slice)):
            s0 = selectors
            s1 = tuple()
        else:
            s0 = selectors[0]
            s1 = selectors[1:]

        if isinstance(s0, slice):
            s0 = self._convert_slice(s0)
        else:
            s0 = self._convert_index(s0)

        return self._data.__getitem__((s0,) + s1)

    def __setitem__(self, selectors, value):
        """Write the data at the requested index and at an duplicated location"""
        if isinstance(selectors, (int, slice)):
            s0 = selectors
            s1 = tuple()
        else:
            s0 = selectors[0]
            s1 = selectors[1:]

        if isinstance(s0, slice):
            write_to = self._convert_slice(s0)
            duplicate_to= self._duplicated_slice(write_to)
        else:
            write_to = self._convert_index(s0)
            duplicate_to = self._duplicated_index(write_to)

        self._data.__setitem__((write_to,) + s1, value)
        self._data.__setitem__((duplicate_to,) + s1, value)


def hhmmss(t: float, dec=0):
    """Format time in seconds to form hh:mm:ss

    Arguments
    ---------
    t : float
        Timestamp in seconds
    dec : int (default 0)
        number of decimal places to show for the seconds
    """
    h = int(t / 3600)
    t -= h * 3600
    m = int(t / 60)
    t -= m * 60
    s = t
    template = "{{}}:{{:02d}}:{{:0{}.0{}f}}".format(dec + 2, dec)
    return template.format(h, m, s)
