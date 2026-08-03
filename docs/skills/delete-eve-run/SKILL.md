---
name: delete-eve-run
description: "Safely delete one or more local EvE run directories, including runs whose read-only files or directories make rm fail. Use when the user explicitly asks to remove run roots under .runs/eve/, or when rm reports Permission denied or Read-only file system while pruning an EvE run."
---

# Delete an EvE Run

Delete only run roots explicitly named by the user. Treat this as destructive and irreversible.

## Delegate the Deletion

- Always use a subagent to execute the permission changes, deletion, and final existence checks. This isolates destructive command output from the main conversation.
- Before delegating, have the main agent validate the user's deletion authorization and every exact absolute run path.
- Give the subagent only the validated paths, steps 3–6 below, and the instruction not to discover, select, or delete any additional run.
- Require a concise per-path result: `DELETED`, `ALREADY_ABSENT`, or `STILL_EXISTS`, plus only the final relevant error when unsuccessful.
- Keep the main agent responsible for reporting the outcome to the user.
- If no subagent facility or concurrency slot is available, stop and report that deletion could not be delegated. Do not fall back to deleting from the main agent.

## Workflow

1. Resolve every requested path and verify that it is a specific run directory below the repository's `.runs/eve/` tree. Reject `.runs/eve`, an app directory, an empty value, `/`, paths outside the repository, and paths containing unresolved variables or globs.
2. Check whether each path exists. Report already-absent paths instead of treating them as failures.
3. Check for an active EvE process that references the exact run path. If one exists, do not delete the live run without explicit user confirmation to stop or delete it.
4. Make the run owner-writable, then delete it:

   ```bash
   chmod -R u+rwX -- "$RUN_ROOT"
   rm -rf -- "$RUN_ROOT"
   ```

   Use `chmod`, not `chomd`. `u+rwX` adds read/write permission for the owner and execute permission only to directories or already-executable files.

5. If `chmod` fails because the sandbox exposes the filesystem as read-only, retry the same narrowly scoped `chmod` and `rm` commands with the environment's escalation mechanism. Do not broaden the path or use a wildcard.
6. Verify completion with an existence check for every exact path:

   ```bash
   if [ -e "$RUN_ROOT" ]; then
     echo "STILL_EXISTS $RUN_ROOT"
   else
     echo "DELETED $RUN_ROOT"
   fi
   ```

7. Report each run as deleted, already absent, or still present. Never claim success solely from `rm` producing no output.

## Guardrails

- Require an explicit deletion request; inspection, debugging, or status requests do not authorize deletion.
- Quote paths and pass `--` before path arguments.
- Never use globbing, command substitution, or a partially constructed path with `rm -rf`.
- Preserve unrelated runs and files.
- Do not use `sudo`, change ownership, or remount a filesystem unless the user separately authorizes that broader action.
