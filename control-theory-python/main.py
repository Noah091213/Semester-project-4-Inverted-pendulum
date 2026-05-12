#include <open62541pp/open62541pp.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>

struct PendulumRow {
    int32_t angle = 0;
    int32_t position = 0;
    int32_t velocity = 0;
    int32_t control = 0;
};

// Map to align data: Key is the 64-bit PLC Timestamp
std::map<int64_t, PendulumRow> log_buffer;

int main() {
    try {
        opcxx::Client client;
        client.connect("opc.tcp://127.0.0.1:4841");

        // 1. Create a subscription (5ms requested interval)
        auto sub = client.createSubscription(std::chrono::milliseconds(5));

        // 2. Helper to add monitored items
        auto monitor = [&](const std::string& nodePath, const std::string& colName) {
            // Using the node path structure from your previous B&R snippet
            opcxx::NodeId nodeId(1, nodePath); 
            
            sub->subscribeDataChange(nodeId, [colName](const opcxx::DataValue& dv) {
                if (!dv.hasValue()) return;

                int64_t ts = dv.sourceTimestamp().count(); // Raw PLC timestamp
                int32_t val = dv.value().get<int32_t>();    // Automatic signed int handling

                if (colName == "Angle") log_buffer[ts].angle = val;
                else if (colName == "Pos") log_buffer[ts].position = val;
                else if (colName == "Vel") log_buffer[ts].velocity = val;
                else if (colName == "Ctrl") log_buffer[ts].control = val;
            });
        };

        // Define your 4 B&R variables
        monitor("Program:Angle", "Angle");
        monitor("Program:Position", "Pos");
        monitor("Program:Velocity", "Vel");
        monitor("Program:Control", "Ctrl");

        std::cout << "Recording for 20 seconds..." << std::endl;
        
        // Use a simple loop to keep the client alive
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < std::chrono::seconds(20)) {
            client.runIterate(); // Processes the OPC UA stack
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
        }

        // 3. The "StringIO" equivalent: std::stringstream
        std::stringstream csv;
        csv << "Timestamp,Angle,Position,Velocity,Control\n";

        for (auto const& [ts, data] : log_buffer) {
            csv << ts << "," << data.angle << "," << data.position << "," 
                << data.velocity << "," << data.control << "\n";
        }

        // 4. Final Write to Disk
        std::ofstream file("pendulum_results.csv");
        file << csv.str();
        file.close();

        std::cout << "Saved " << log_buffer.size() << " aligned rows to CSV." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}