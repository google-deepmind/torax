# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Once you have signed the CLA and your code is ready for review we encourage you
to use the "ready-for-review" tag which will highlight to us that a review is
required.

Please also squash your commits once you have completed your changes into a
single commit. This helps us keep our commit history clean and easy to parse.

### Contributing tips

- **Keep pull requests small and focused:** Please keep your pull requests (PRs)
  as simple and short as possible, ideally addressing a single issue. This
  reduces the time it takes for us to review changes and makes it easier to
  give high-quality, directed feedback in line with TORAX development patterns.
  Larger features can be comprised of multiple smaller chained PRs, and feel
  free to discuss with us how to structure changes.

- **"Domain expertise required" issues:** Many of our issues have the "Domain
  expertise required" label. We typically do not accept PRs on these issues from
  contributors who are not domain experts.

- **Discuss design for other issues:** When looking to handle other issues,
  please reach out and start a discussion on how a feature should be designed.
  This helps make sure changes are made that are compatible with design patterns
  in the rest of the codebase and are compatible with any future development
  plans we may have.

- **Use of GenAI:** When using LLMs to generate code to address our issues,
  please keep in mind the above points, it is important that the generated code
  is understood by the developer and tests are properly run and PRs maintain
  focus. Please keep in mind that reviewing is time-consuming and PRs that do
  not are likely to have longer review times.

You can find more contribution tips in our
[documentation](https://torax.readthedocs.io/en/latest/contribution_tips.html#contribution-tips).

### Additional info

It may also be useful to know that we have an internal repository that is the
SSOT for our codebase. We use a tool called
[copybara](https://github.com/google/copybara) to keep consistency between our
internal repo and the external GitHub repository. The typical workflow we follow
with PRs is as follows:

- PR written and commits squashed by author and reviewed by TORAX devs.

- We trigger a copybara import to the internal repo.

- Some further minor changes may be done in internal review

- Submission to our internal repo triggers a commit to GitHub that copybara
  relates to the original PR and "merges" the PR. The actual commit may have
  slightly different code to the final state of the PR due to any internal
  changes. Author attribution is maintained in this process.
