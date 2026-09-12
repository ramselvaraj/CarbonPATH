# Remote Machine Access

The CarbonPATH execution machine is available through SSH at `10.218.101.21`.

## SSH Connection

The local SSH alias is `carbonpath-remote` and uses the remote user `rselvar8`.

```bash
ssh carbonpath-remote
```

SSH will prompt for the account password. The password is not stored in the
repository or SSH configuration.

For commands that need to run from this environment, an authenticated shared
connection can be opened for 10 minutes:

```bash
ssh -fN carbonpath-remote
```

## SFTP

Use the same alias for file transfers:

```bash
sftp carbonpath-remote
```

The remote project directory is:

```text
~/work/CarbonPATH
```

## Python Environment

On the remote machine:

```bash
cd ~/work/CarbonPATH
source carbonpath/bin/activate
```

The environment uses Python 3.12 and dependencies from `requirements.txt`.

Commands can also use the environment directly without activation:

```bash
cd ~/work/CarbonPATH
carbonpath/bin/python -m main --workload 1 --run_mode run_sim_anneal --cost_profile t1
```

Run the test suite with:

```bash
carbonpath/bin/python -m unittest discover -s tests -v
```

## Updating Code

Code is maintained in GitHub and pulled onto the remote machine:

```bash
cd ~/work/CarbonPATH
git pull --ff-only origin main
```

Experiment outputs, caches, calibration results, and the local Python
environment should remain on the remote machine and should not be committed.
