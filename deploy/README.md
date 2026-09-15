# Deploying the collector

The live aisstream collector is the one component with a deadline attached.
PortWatch and price history can be pulled at any time; **raw AIS cannot**. There
is no backfill, so every hour the collector is not running is an hour that is
gone permanently. Get this running before doing any analysis work.

## What you need

A small always-on Linux box — a cheap VPS or a Raspberry Pi is plenty. The
collector is I/O-bound, not CPU-bound: it decodes JSON and appends Parquet.

Rough sizing for the three zones enabled by default (Hormuz, Bab el-Mandeb,
Suez): well under 100 MB of RAM, and disk growth in the low hundreds of MB per
month. Measure it in your first week rather than trusting that estimate — traffic
volume through these boxes varies enormously with the political situation, which
is rather the point of the project.

## Install

```bash
# 1. Service user and layout
sudo useradd --system --home /opt/tanker-tape --shell /usr/sbin/nologin tankertape
sudo mkdir -p /opt/tanker-tape /etc/tanker-tape
sudo chown tankertape:tankertape /opt/tanker-tape

# 2. Code and dependencies
sudo -u tankertape git clone https://github.com/minqz123/tanker-tape.git /opt/tanker-tape
cd /opt/tanker-tape
# The collector does not need the research or dashboard extras.
sudo -u tankertape uv sync

# 3. Secrets - the collector needs only the aisstream key
sudo tee /etc/tanker-tape/collector.env >/dev/null <<'EOF'
AISSTREAM_API_KEY=your-key-here
TANKER_TAPE_DATA_DIR=/opt/tanker-tape/data
TANKER_TAPE_LOG_LEVEL=INFO
EOF
sudo chmod 600 /etc/tanker-tape/collector.env
sudo chown tankertape:tankertape /etc/tanker-tape/collector.env

# 4. Check the geometry BEFORE starting - see the warning below
sudo -u tankertape /opt/tanker-tape/.venv/bin/tanker-tape validate-gates

# 5. Start it
sudo cp deploy/tanker-tape-collector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tanker-tape-collector
```

### Check the bounding boxes first

`validate-gates` prints the aisstream bounding box for every zone as
`[latitude, longitude]` pairs. Confirm they cover the water you expect. A
latitude/longitude swap does not raise an error — it subscribes to an empty
patch of ocean and returns nothing, and you will not notice until you look at
the data days later. Those days are unrecoverable, which is why this check comes
before `systemctl enable`.

## Confirm it is actually collecting

```bash
systemctl status tanker-tape-collector
journalctl -u tanker-tape-collector -f
```

Within about five minutes of starting you should see a flush line:

```
stage=aisstream.flush partition=2026-09-15/19 positions=8412 static=137 received=...
```

`positions=0` on repeated flushes means the socket is up but nothing matches
your boxes — check the geometry, then check whether terrestrial receiver coverage
reaches that area at all. A connected collector returning nothing looks identical
to a healthy one in `systemctl status`, so watch the flush counts, not the unit
state.

Then confirm files are landing:

```bash
find /opt/tanker-tape/data/raw/ais_positions -name '*.parquet' | tail
```

## Monitoring

The failure mode to design against is silent: the unit is `active (running)`,
the websocket is connected, and no useful rows are arriving. Watch the flush
counters rather than the process.

Worth alerting on:

- No new Parquet file under `data/raw/ais_positions` in the last hour.
- `dropped_queue_full` climbing in the flush lines — the writer is not keeping up.
  Lower `flush_seconds` or raise `max_queue` in `AisCollector`.
- `reconnects` climbing steadily rather than occasionally.
- Disk usage on the data partition.

A crude but effective check, run from cron:

```bash
find /opt/tanker-tape/data/raw/ais_positions -name '*.parquet' -mmin -90 \
  | grep -q . || echo "tanker-tape: no AIS data written in 90 minutes"
```

## Logs

Journald handles rotation. To keep the unit's logs bounded:

```bash
sudo journalctl --vacuum-time=30d
```

## Upgrading

```bash
cd /opt/tanker-tape
sudo -u tankertape git pull
sudo -u tankertape uv sync
sudo systemctl restart tanker-tape-collector
```

The restart sends SIGTERM, which flushes buffered rows before exit, so an
upgrade costs a few seconds of collection rather than the current buffer.

## Backups

The AIS data under `data/raw/` is the irreplaceable part of this project —
everything else can be re-fetched from a public API. Back it up somewhere off
the box. A weekly `rsync` or `rclone` of `data/raw/ais_positions` and
`data/raw/ais_static` to object storage is enough.
