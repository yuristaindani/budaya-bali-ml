// "use server";

// export const postChat = async (question: string) => {
//   try {
//     // Gunakan URL lengkap dengan http://
//     const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    
//     console.log(`Mengirim request ke: ${apiUrl}/ask`); // Untuk debugging
    
//     const response = await fetch(`${apiUrl}/ask`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ question }),
//     });

//     if (!response.ok) {
//       const errorText = await response.text();
//       console.error(`Error response: ${errorText}`);
//       throw new Error(`Request failed with status ${response.status}`);
//     }

//     const data = await response.json();
//     return {
//       success: true,
//       data: {
//         content: data.answer,
//         sources: data.sources || [],
//       },
//     };
//   } catch (error) {
//     console.error('Error details:', error);
//     return {
//       success: false,
//       message: "Gagal terhubung ke server. Pastikan backend sedang berjalan.",
//       error: error instanceof Error ? error.message : String(error),
//     };
//   }
// };

// "use server";

// export const postChat = async (question: string) => {
//   try {
//     const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
//     const response = await fetch(`${apiUrl}/ask`, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ question }),
//     });

//     const data = await response.json();
//     return {
//       success: true,
//       data: {
//         content: data.answer,
//         sources: data.sources || [],  // All sources
//         mainImage: data.main_image    // First image only
//       },
//     };
//   } catch (error) {
//     return {
//       success: false,
//       message: "Error occurred while fetching response",
//     };
//   }
// };

"use server";

export const postChat = async (question: string) => {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${apiUrl}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      return { 
        success: false, 
        message: "Failed to get response" 
      };
    }

    const data = await response.json();

    return {
      success: data.success,
      message: "Success",
      data: {
        content: data.answer,
        sources: data.sources || [],
        language: data.detected_language || "en"
      },
    };
  } catch (error) {
    return { 
      success: false, 
      message: "Error processing request" 
    };
  }
};