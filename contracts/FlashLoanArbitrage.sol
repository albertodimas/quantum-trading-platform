// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@aave/core-v3/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "@aave/core-v3/contracts/interfaces/IPoolAddressesProvider.sol";
import "@aave/core-v3/contracts/dependencies/openzeppelin/contracts/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
import "@uniswap/v2-periphery/contracts/interfaces/IUniswapV2Router02.sol";

/**
 * @title FlashLoanArbitrage
 * @dev Contrato para ejecutar arbitraje usando flash loans de Aave
 * @author Quantum Trading Platform
 */
contract FlashLoanArbitrage is FlashLoanSimpleReceiverBase, Ownable {
    
    // Eventos
    event ArbitrageExecuted(
        address indexed asset,
        uint256 amount,
        uint256 profit,
        address indexed executor
    );
    
    event FlashLoanExecuted(
        address indexed asset,
        uint256 amount,
        uint256 premium
    );
    
    // Estructuras
    struct ArbitrageParams {
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        address dexA;
        address dexB;
        bytes swapDataA;
        bytes swapDataB;
        uint256 minProfitThreshold;
    }
    
    struct DEXInfo {
        address router;
        uint8 dexType; // 0 = Uniswap V2, 1 = Uniswap V3, 2 = SushiSwap, etc.
        bool isActive;
    }
    
    // Variables de estado
    mapping(address => DEXInfo) public supportedDEXs;
    mapping(address => bool) public authorizedCallers;
    
    // Constantes
    uint256 private constant MAX_INT = 2**256 - 1;
    uint256 private constant PROFIT_THRESHOLD = 1e15; // 0.001 ETH mínimo
    
    // Modificadores
    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender] || msg.sender == owner(), "Not authorized");
        _;
    }
    
    constructor(IPoolAddressesProvider _addressProvider) 
        FlashLoanSimpleReceiverBase(_addressProvider) 
    {
        // Autorizar al owner por defecto
        authorizedCallers[msg.sender] = true;
        
        // Configurar DEXs soportados inicialmente
        _setupInitialDEXs();
    }
    
    /**
     * @dev Ejecutar arbitraje con flash loan
     * @param asset El activo para el flash loan
     * @param amount La cantidad del flash loan
     * @param params Parámetros del arbitraje
     */
    function executeArbitrage(
        address asset,
        uint256 amount,
        ArbitrageParams calldata params
    ) external onlyAuthorized {
        require(amount > 0, "Amount must be greater than 0");
        require(supportedDEXs[params.dexA].isActive, "DEX A not supported");
        require(supportedDEXs[params.dexB].isActive, "DEX B not supported");
        
        // Codificar parámetros para el callback
        bytes memory data = abi.encode(params);
        
        // Iniciar flash loan
        POOL.flashLoanSimple(
            address(this),
            asset,
            amount,
            data,
            0 // referralCode
        );
    }
    
    /**
     * @dev Callback ejecutado por Aave después del flash loan
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        require(msg.sender == address(POOL), "Caller must be POOL");
        require(initiator == address(this), "Initiator must be this contract");
        
        // Decodificar parámetros
        ArbitrageParams memory arbitrageParams = abi.decode(params, (ArbitrageParams));
        
        // Ejecutar estrategia de arbitraje
        uint256 profit = _executeArbitrageStrategy(asset, amount, arbitrageParams);
        
        // Verificar que hay suficiente profit para pagar el flash loan + premium
        uint256 totalDebt = amount + premium;
        require(IERC20(asset).balanceOf(address(this)) >= totalDebt, "Insufficient funds to repay");
        require(profit >= arbitrageParams.minProfitThreshold, "Profit below threshold");
        
        // Aprobar el repago del flash loan
        IERC20(asset).approve(address(POOL), totalDebt);
        
        emit FlashLoanExecuted(asset, amount, premium);
        emit ArbitrageExecuted(asset, amount, profit, tx.origin);
        
        return true;
    }
    
    /**
     * @dev Ejecutar la estrategia de arbitraje
     */
    function _executeArbitrageStrategy(
        address asset,
        uint256 amount,
        ArbitrageParams memory params
    ) internal returns (uint256 profit) {
        uint256 initialBalance = IERC20(asset).balanceOf(address(this));
        
        // Paso 1: Intercambiar en DEX A (comprar barato)
        uint256 amountOut1 = _swapOnDEX(
            params.dexA,
            params.tokenIn,
            params.tokenOut,
            amount,
            params.swapDataA
        );
        
        // Paso 2: Intercambiar en DEX B (vender caro)
        uint256 amountOut2 = _swapOnDEX(
            params.dexB,
            params.tokenOut,
            params.tokenIn,
            amountOut1,
            params.swapDataB
        );
        
        uint256 finalBalance = IERC20(asset).balanceOf(address(this));
        
        // Calcular profit
        profit = finalBalance > initialBalance ? finalBalance - initialBalance : 0;
        
        return profit;
    }
    
    /**
     * @dev Ejecutar swap en un DEX específico
     */
    function _swapOnDEX(
        address dex,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        bytes memory swapData
    ) internal returns (uint256 amountOut) {
        DEXInfo memory dexInfo = supportedDEXs[dex];
        require(dexInfo.isActive, "DEX not active");
        
        // Aprobar tokens para el router
        IERC20(tokenIn).approve(dexInfo.router, amountIn);
        
        if (dexInfo.dexType == 0) {
            // Uniswap V2 style
            amountOut = _swapUniswapV2(dexInfo.router, tokenIn, tokenOut, amountIn);
        } else if (dexInfo.dexType == 1) {
            // Uniswap V3 style
            amountOut = _swapUniswapV3(dexInfo.router, tokenIn, tokenOut, amountIn, swapData);
        } else {
            revert("Unsupported DEX type");
        }
        
        return amountOut;
    }
    
    /**
     * @dev Swap en Uniswap V2
     */
    function _swapUniswapV2(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256 amountOut) {
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        uint256[] memory amounts = IUniswapV2Router02(router).swapExactTokensForTokens(
            amountIn,
            0, // amountOutMin (en producción usar cálculo real)
            path,
            address(this),
            block.timestamp + 300
        );
        
        return amounts[amounts.length - 1];
    }
    
    /**
     * @dev Swap en Uniswap V3
     */
    function _swapUniswapV3(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        bytes memory swapData
    ) internal returns (uint256 amountOut) {
        // Decodificar parámetros específicos de V3
        uint24 fee = abi.decode(swapData, (uint24));
        
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: address(this),
            deadline: block.timestamp + 300,
            amountIn: amountIn,
            amountOutMinimum: 0, // En producción usar cálculo real
            sqrtPriceLimitX96: 0
        });
        
        amountOut = ISwapRouter(router).exactInputSingle(params);
        return amountOut;
    }
    
    /**
     * @dev Configurar DEXs iniciales
     */
    function _setupInitialDEXs() internal {
        // Uniswap V2
        supportedDEXs[0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D] = DEXInfo({
            router: 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D,
            dexType: 0,
            isActive: true
        });
        
        // Uniswap V3
        supportedDEXs[0xE592427A0AEce92De3Edee1F18E0157C05861564] = DEXInfo({
            router: 0xE592427A0AEce92De3Edee1F18E0157C05861564,
            dexType: 1,
            isActive: true
        });
        
        // SushiSwap
        supportedDEXs[0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F] = DEXInfo({
            router: 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F,
            dexType: 0,
            isActive: true
        });
    }
    
    /**
     * @dev Agregar soporte para nuevo DEX
     */
    function addDEX(
        address dexAddress,
        address router,
        uint8 dexType,
        bool isActive
    ) external onlyOwner {
        supportedDEXs[dexAddress] = DEXInfo({
            router: router,
            dexType: dexType,
            isActive: isActive
        });
    }
    
    /**
     * @dev Autorizar/desautorizar caller
     */
    function setAuthorizedCaller(address caller, bool authorized) external onlyOwner {
        authorizedCallers[caller] = authorized;
    }
    
    /**
     * @dev Activar/desactivar DEX
     */
    function setDEXStatus(address dex, bool isActive) external onlyOwner {
        supportedDEXs[dex].isActive = isActive;
    }
    
    /**
     * @dev Retirar tokens en caso de emergencia
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).transfer(owner(), amount);
    }
    
    /**
     * @dev Retirar ETH en caso de emergencia
     */
    function emergencyWithdrawETH() external onlyOwner {
        payable(owner()).transfer(address(this).balance);
    }
    
    /**
     * @dev Función para recibir ETH
     */
    receive() external payable {}
    
    /**
     * @dev Función fallback
     */
    fallback() external payable {}
    
    /**
     * @dev Obtener información de un DEX
     */
    function getDEXInfo(address dex) external view returns (DEXInfo memory) {
        return supportedDEXs[dex];
    }
    
    /**
     * @dev Verificar si un caller está autorizado
     */
    function isAuthorized(address caller) external view returns (bool) {
        return authorizedCallers[caller] || caller == owner();
    }
}