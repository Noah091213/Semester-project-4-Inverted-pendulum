import asyncio
import csv
import time
from asyncua import Client, ua

url = "opc.tcp://192.168.8.101:4841"
namespace = "urn:B&R/pv/"
DURATION = 30  # Total recording time
SAMPLE_RATE = 0.001  # 5ms loop timer
timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = r"G:\My Drive\Uddannelse\SDU\Semester-project-4-Inverted-pendulum"
filename = f"pendulum_data_{timestamp_suffix}.csv"
final_filepath = os.path.join(base_dir, filename)

async def main():
    print(f"Connecting to {url} ...")
    async with Client(url=url, watchdog_intervall=5) as client:
        nsidx = await client.get_namespace_index(namespace)
        
        # Get all 4 nodes
        nodes = {
            "State": await client.nodes.root.get_child(f"0:Objects/4:PLC/6:Modules/6:::/6:Program/6:opc_state"),
            "Step": await client.nodes.root.get_child(f"0:Objects/4:PLC/6:Modules/6:::/6:Program/6:opc_step"),
            "X": await client.nodes.root.get_child(f"0:Objects/4:PLC/6:Modules/6:::/6:Program/6:opc_x"),
            "Theta": await client.nodes.root.get_child(f"0:Objects/4:PLC/6:Modules/6:::/6:Program/6:opc_theta"),
            "U": await client.nodes.root.get_child(f"0:Objects/4:PLC/6:Modules/6:::/6:Program/6:opc_u")
        }

        data_rows = []
        start_time = time.time()
        end_time = start_time + DURATION

        print(f"Recording for {DURATION} seconds...")
        
        while time.time() < end_time:
            loop_start = time.time()
            
            # Read values directly from the PLC
            row = {}
            
            # Perform the reads
            # Note: client.read_values(list_of_nodes) is faster than individual calls
            values = await client.read_values(list(nodes.values()))
            
            for name, val in zip(nodes.keys(), values):
                row[name] = val
            
            data_rows.append(row)

            # Control the loop frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0, SAMPLE_RATE - elapsed)
            await asyncio.sleep(sleep_time)

        # Save to CSV
        print(f"Saving {len(data_rows)} rows to {filename}...")
        with open(final_filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(nodes.keys()))
            writer.writeheader()
            writer.writerows(data_rows)

        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())