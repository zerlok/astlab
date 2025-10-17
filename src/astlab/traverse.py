import typing as t
from collections import deque

T = t.TypeVar("T", bound=t.Hashable)


def traverse_dfs_post_order(root: T, children: t.Callable[[T], t.Iterable[T]]) -> t.Iterable[T]:
    stack = deque[tuple[T, bool]]([(root, False)])

    while stack:
        item, processed = stack.pop()
        if processed:
            yield item

        else:
            stack.append((item, True))

            for child in children(item):
                stack.append((child, False))
