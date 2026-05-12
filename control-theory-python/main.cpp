#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>
#include "open62541.h"

struct PendulumRow {
    int32_t angle = 0, position = 0, velocity = 0, control = 0;
};

// Key = Raw PLC Timestamp, Value = Row data
std::map<UA_DateTime, PendulumRow> log_buffer;

static void onDataChange(UA_Client *client, UA_UInt32 subId, void *subContext,
                         UA_UInt32 monId, void *monContext, UA_DataValue *value) {
    
    // Check if the value is valid and is an Int32 (signed integer)
    if (value->hasValue && UA_Variant_isScalar(&value->value) && 
        value->value.type == &UA_TYPES[UA_TYPES_INT32]) {
        
        int32_t val = *(UA_Int32*)value->value.data;
        std::string col = (char*)monContext;
        
        // Use the SourceTimestamp from the PLC for 5ms alignment
        UA_DateTime ts = value->sourceTimestamp;
        
        if (col == "Angle") log_buffer[ts].angle = val;
        else if (col == "Pos") log_buffer[ts].position = val;
        else if (col == "Vel") log_buffer[ts].velocity = val;
        else if (col == "Ctrl") log_buffer[ts].control = val;
    }
}

int main() {
    UA_Client *client = UA_Client_new();
    UA_ClientConfig_setDefault(UA_Client_getConfig(client));

    if (UA_Client_connect(client, "opc.tcp://127.0.0.1:4841") != UA_STATUSCODE_GOOD) {
        UA_Client_delete(client);
        return 1;
    }

    UA_CreateSubscriptionRequest request = UA_CreateSubscriptionRequest_default();
    request.requestedPublishingInterval = 5.0; // 5ms
    UA_CreateSubscriptionResponse response = UA_Client_Subscriptions_create(client, request, NULL, NULL, NULL);

    auto add_mon = [&](const char* id, const char* name) {
        // Use namespace 6 for B&R PVs or 4 depending on your configuration
        UA_NodeId node = UA_NODEID_STRING(6, (char*)id); 
        UA_MonitoredItemCreateRequest mon = UA_MonitoredItemCreateRequest_default(node);
        mon.requestedParameters.samplingInterval = 5.0;

        // UPDATED FUNCTION NAME FOR v1.4
        UA_Client_MonitoredItems_createDataChange(
            client, response.subscriptionId,
            UA_TIMESTAMPSTORETURN_BOTH, mon,
            (void*)name, onDataChange, NULL);
    };

    add_mon("::Program:Angle", "Angle");
    add_mon("::Program:Position", "Pos");
    add_mon("::Program:Velocity", "Vel");
    add_mon("::Program:Control", "Ctrl");

    std::cout << "Recording... (20s)" << std::endl;
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(20)) {
        UA_Client_run_iterate(client, 1);
    }

    // StringIO equivalent: Memory buffer to CSV
    std::stringstream ss;
    ss << "Timestamp,Angle,Position,Velocity,Control\n";
    for (auto const& [ts, d] : log_buffer) {
        ss << ts << "," << d.angle << "," << d.position << "," << d.velocity << "," << d.control << "\n";
    }

    std::ofstream f("pendulum_results.csv");
    f << ss.str();

    UA_Client_disconnect(client);
    UA_Client_delete(client);
    return 0;
}