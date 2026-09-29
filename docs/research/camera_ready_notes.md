# Camera-ready notes (not part of the anonymous submission)

Issue #235 (submission housekeeping) moved these two comment blocks out of
`overleaf_drafts/acl_latex.tex`: under `[review]` they printed nothing, but
they named the authors, their affiliations and the compute allocation in the
submission sources. Restore them when the class option becomes `[final]` or
`[preprint]`.

## Author block (replaces the placeholder `\author` in the preamble)

```latex
% ---------------------------------------------------------------------------
% CAMERA-READY AUTHOR BLOCK (commented out; do NOT enable under [review]).
% Under [review] acl.sty prints "Anonymous ACL submission" whatever \author
% holds, so the placeholder above is what the anonymous PDF needs. When the
% class option becomes [final] or [preprint], replace the placeholder with
% the block below after resolving every TODO.
%
% TODO(camera-ready): author ORDER is not recorded anywhere in the repo. The
%   order below is only the order in which issue #173 names people; Ian to
%   confirm with Prof. Siddique and Prof. Firey.
% TODO(camera-ready): James Wong attends the paper meetings and reviews the
%   attribution and related-work sections (docs/meetings/meeting_20260914.md)
%   but issue #173 does not list him as an author. Confirm whether he is an
%   author; his line is left commented a second time.
% TODO(camera-ready): affiliations and e-mail addresses. The only affiliation
%   on record is Prof. Firey's (issue #173). Do not guess the others.
%
% \author{Ian Rowe Rojas \\
%   TODO affiliation \\
%   \texttt{TODO email} \\\And
%   Abigail Firey \\
%   Department of History \\
%   University of Kentucky \\
%   \texttt{TODO email} \\\And
%   A. B. Siddique \\
%   TODO affiliation \\
%   \texttt{TODO email} \\}
% %  \And James Wong \\ TODO affiliation \\ \texttt{TODO email}
% ---------------------------------------------------------------------------
```

## Acknowledgments (before `\bibliography{custom}`)

```latex
% ---------------------------------------------------------------------------
% ACKNOWLEDGMENTS placeholder (CAMERA-READY ONLY). ACL/ARR: omit under
% [review], restore under [final] or [preprint]. Uncomment the block below
% when de-anonymising, and at the same time:
%   * replace the \author placeholder with the camera-ready block kept in
%     comments above \begin{document};
%   * Appendix A: drop the "unpublished documentation and correspondence"
%     footnote (its source is then an author) and name the director where the text says "the
%     project's director".
%
% \section*{Acknowledgments}
% The corpus in this paper is drawn from the Carolingian Canon Law project
% (CCL), which Abigail Firey, a co-author of this paper, directs at the
% University of Kentucky. We thank the CCL transcribers and proofreaders, credited individually in the
% CCL interface, whose to-the-letter work makes this benchmark possible, the
% graduate students who assigned source keys under the director's
% supervision, and the CCL developer for the TEI-P5 exports.
% TODO(camera-ready): name the CCL developer with his consent and preferred
% form of name; confirm with Prof. Firey who else must be thanked.
% TODO(camera-ready): funding and compute acknowledgements (NCSA Delta
% allocation, any grant numbers) are not recorded in this repo; ask Ian.
%
% ---------------------------------------------------------------------------
```
