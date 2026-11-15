"""Connect a Bluetooth heart-rate tracker to the dashboard.

    python tracker_ble.py --list                              # show nearby heart-rate devices
    python tracker_ble.py --astronaut "Asha Verma"            # use the first one found
    python tracker_ble.py --astronaut "Leo Brandt" --name Polar

Works with any device that speaks the standard Bluetooth "Heart Rate" profile:
chest straps and arm bands generally do, and many sports watches can once you
switch on their "broadcast heart rate" mode. Devices that only talk to their
own phone app will not show up; send their data through tracker_server.py
instead.

Needs:  pip install bleak
The reading appears on the dashboard as that astronaut's live heart rate.
Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import live_store
import simulator

HEART_RATE_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"


def parse_heart_rate(data: bytes) -> tuple[int, list[float]]:
    """Decode a Heart Rate Measurement packet (Bluetooth characteristic 0x2A37).

    Returns (beats per minute, [intervals between beats in milliseconds]).
    Byte 0 is a set of flags:
        bit 0  heart rate is 16-bit (otherwise 8-bit)
        bit 3  a 16-bit "energy expended" field follows the heart rate
        bit 4  one or more 16-bit beat-to-beat (RR) intervals follow, in 1/1024 s
    """
    if len(data) < 2:
        raise ValueError("packet too short")
    flags = data[0]
    if flags & 0x01:
        bpm, pos = int.from_bytes(data[1:3], "little"), 3
    else:
        bpm, pos = data[1], 2
    if flags & 0x08:
        pos += 2
    rr_ms = []
    if flags & 0x10:
        while pos + 1 < len(data):
            rr_ms.append(int.from_bytes(data[pos:pos + 2], "little") * 1000 / 1024)
            pos += 2
    return bpm, rr_ms


async def find_devices(timeout: float = 8.0) -> list:
    """Nearby devices advertising the heart-rate service, strongest signal first."""
    from bleak import BleakScanner

    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    matches = [(device, adv) for device, adv in found.values()
               if HEART_RATE_SERVICE in [u.lower() for u in (adv.service_uuids or [])]]
    return sorted(matches, key=lambda pair: -(pair[1].rssi or -999))


async def stream(astronaut: str, name_filter: str | None) -> None:
    from bleak import BleakClient

    while True:
        print("Scanning for heart-rate devices ...")
        matches = await find_devices()
        if name_filter:
            matches = [m for m in matches if name_filter.lower() in (m[0].name or "").lower()]
        if not matches:
            print("  none found. Is the tracker on, worn, and in broadcast mode? Retrying in 5 s.")
            await asyncio.sleep(5)
            continue
        device = matches[0][0]
        source = device.name or device.address
        print(f"Connecting to {source} ...")
        disconnected = asyncio.Event()

        def on_packet(_characteristic, data: bytearray) -> None:
            try:
                bpm, _rr = parse_heart_rate(bytes(data))
            except ValueError:
                return
            if bpm > 0:   # 0 means the sensor has no skin contact yet
                live_store.write(astronaut, {"heart_rate": bpm}, source=source)
                print(f"  {astronaut}: {bpm} bpm   ", end="\r", flush=True)

        try:
            async with BleakClient(device, disconnected_callback=lambda _client: disconnected.set()) as client:
                await client.start_notify(HEART_RATE_MEASUREMENT, on_packet)
                print(f"Connected. Sending {source} heart rate to the dashboard as {astronaut}.")
                await disconnected.wait()
        except Exception as exc:  # noqa: BLE001 - Bluetooth errors vary by operating system
            print(f"\nConnection problem: {exc}")
        print("\nDisconnected. Reconnecting in 3 s ...")
        await asyncio.sleep(3)


def main() -> None:
    names = [a.name for a in simulator.CREW]
    parser = argparse.ArgumentParser(description="Send a Bluetooth heart-rate tracker's readings to the dashboard")
    parser.add_argument("--astronaut", choices=names, help="whose heart rate this is")
    parser.add_argument("--name", help="only connect to a device whose name contains this text")
    parser.add_argument("--list", action="store_true", help="list nearby heart-rate devices and exit")
    args = parser.parse_args()
    try:
        import bleak  # noqa: F401
    except ImportError:
        sys.exit("This needs the Bluetooth library. Install it with:  pip install bleak")

    if args.list:
        try:
            matches = asyncio.run(find_devices())
        except Exception as exc:  # noqa: BLE001 - Bluetooth errors vary by operating system
            sys.exit(f"Bluetooth is not available on this computer ({exc}).\n"
                     "Check that Bluetooth is switched on and that the terminal is allowed to use it.")
        if not matches:
            print("No heart-rate devices found.")
        for device, adv in matches:
            print(f"  {device.name or '(no name)':30s} {device.address}   signal {adv.rssi} dBm")
        return
    if not args.astronaut:
        parser.error("--astronaut is required (one of: " + ", ".join(names) + ")")
    try:
        asyncio.run(stream(args.astronaut, args.name))
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Bluetooth is not available on this computer ({exc}).\n"
                 "Check that Bluetooth is switched on and that the terminal is allowed to use it.")


if __name__ == "__main__":
    main()
