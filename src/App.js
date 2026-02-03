import React, { useState } from "react";
import { ethers } from "ethers";
import "./App.css";

function App() {
  

  const [blockNumber, setBlockNumber] = useState(null);
  const [txSent, setTxSent] = useState(null);
  

  // MetaMask provider
  const getMetaMaskProvider = () => {
    if (!window.ethereum) {
      alert("MetaMask not installed");
      return null;
    }
    return new ethers.BrowserProvider(window.ethereum);
  };



  // Get latest block via MetaMask
  const handleButton = async () => {
    try {
      const provider = getMetaMaskProvider();
      if (!provider) return;
      const latestBlock = await provider.getBlockNumber();
      setBlockNumber(latestBlock);
    } catch (err) {
      console.error(err);
    }
  };

  // Send transaction using MetaMask
  const handleSubmitWeb3 = async (e) => {
    e.preventDefault();

    try {
      const data = new FormData(e.target);
      const address = data.get("address");
      const amount = data.get("amount");

      await window.ethereum.request({ method: "eth_requestAccounts" });

      const provider = new ethers.BrowserProvider(window.ethereum);
      const signer = await provider.getSigner();

      const tx = await signer.sendTransaction({
        to: address,
        value: ethers.parseEther(amount),
      });

      setTxSent(`Transaction sent! Tx hash: ${tx.hash}`);
    } catch (err) {
      console.error(err);
      setTxSent("Transaction failed");
    }
  };


  return (
    <div className="App">
      <header className="App-header">
        <h3>Get latest block number</h3>

        
        <button onClick={handleButton}>
          Using MetaMask
        </button>

        <p>Block Number: {blockNumber}</p>

        <hr />

        <h3>Send transaction via MetaMask</h3>
        <form onSubmit={handleSubmitWeb3}>
          <input type="text" name="address" placeholder="Recipient Address" />
          <input type="text" name="amount" placeholder="Amount (ETH)" />
          <input type="submit" value="Send with MetaMask" />
        </form>
        <p>{txSent}</p>

        <hr />

        
        
      </header>
    </div>
  );
}

export default App;
