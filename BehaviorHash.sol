// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract BehaviorHash {
    event BehaviorStored(
        string behaviorId,
        string hashData,
        uint256 timestamp,
        address sender
    );

    mapping(string => string) public hashes;

    function storeBehaviorHash(
        string memory behaviorId,
        string memory hashData,
        uint256 timestamp
    ) public {
        hashes[behaviorId] = hashData;
        emit BehaviorStored(behaviorId, hashData, timestamp, msg.sender);
    }
}
