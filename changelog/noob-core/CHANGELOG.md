# noob-core

<!-- towncrier release notes start -->

## v0.2.1 - 26-10-02

**Fixed**

- [`#267`](https://github.com/miniscope/noob/pull/267) - had a problem where @MarcelMB noticed that running the tube with a trivial graph like mio's display graph when everything is disabled cause the runtime to linearly grow. this was because empty epochs kept growing, and so first_active_epoch had to iterate through more and more things every time. that also causes infinite memory growth, which the scheduler should never have.

  So basically we just don't even store the added sorter when it's trivial, in practice this amounts to checking if the ready set is empty, so it's pretty quick to do, and the rest of the scheduler semantics should remain correct.

## v0.2.0 - 26-09-21

**Docs**

- [`#265`](https://github.com/miniscope/noob/pull/265) - Begin changelog
